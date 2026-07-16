"""
Stage 7 — Serialize the EXACT quantized model chosen by Stage 5 (P0.5/P0.6).

Reads the Stage 5 QDQ checkpoint plus its quantization manifest and exact
artifacts (E2M1 codes + FP8 scales, captured during GPTQ) and emits a packed
checkpoint in the ModelOpt/vLLM W4A16_NVFP4 layout.

Fail-closed contract (P0.5):
  - A complete manifest (quant_artifacts/manifest.json) is REQUIRED. There is
    no "pack all nn.Linear" fallback.
  - Every packed tensor is verified against the on-disk QDQ weight:
        dequantize(codes, scales) == QDQ weight   (bit-exact)
    A mismatch aborts the run (P0.6 invariant).
  - GPT-OSS batched expert slices cannot yet be packed for vLLM (P0.7). If
    the manifest contains expert slices, Stage 7 refuses to run unless
    --allow_hybrid is passed, in which case experts stay BF16, are added to
    the quantization ignore list, and the output is loudly marked HYBRID.

Exact consumption (P0.6):
  - Codes and scales are read from the Stage 5 artifact shards, never
    re-derived from weights. The packed model IS the model GPTQ optimized
    (and the model Stage 6 evaluated, modulo the BF16 QDQ cast).

Output per packed linear (ModelOpt W4A16_NVFP4 layout; the exact contract
against the pinned vLLM is verified in P0.8 / docs/VLLM_NVFP4_CONTRACT.md):
    {name}.weight          uint8        [out, in//2]   packed E2M1 nibble pairs
    {name}.weight_scale    float8_e4m3  [out, in//16]  per-block scales
    {name}.weight_scale_2  bfloat16     [1]            1.0 (W4A16)

Formats other than nvfp4 are not supported by this exporter (the legacy
fp8/int8 re-derivation paths were removed — they violated P0.6 by design).

Usage:
    python stage7_save_modelopt.py \\
        --model_path  models/<stage5-qdq-output> \\
        --output_dir  models/<name>-packed \\
        [--manifest   <model_path>/quant_artifacts/manifest.json] \\
        [--allow_hybrid]

Exit: 0 = success, nonzero = refused/failed (fail closed).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Repo-relative code root (P0.1 fix): the library lives at
# <repo>/opteam-blockwise-gptq regardless of where the repo is checked out.
_CODE_ROOT = _REPO_ROOT / "opteam-blockwise-gptq"
sys.path.insert(0, str(_CODE_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Raw tensor loading (the QDQ weights exactly as Stage 5 saved them)
# ══════════════════════════════════════════════════════════════════════════════

def load_qdq_state_dict(model_path: Path) -> dict:
    """Load every tensor from the Stage 5 safetensors shards, unmodified.

    Deliberately does NOT instantiate the model: from_pretrained with
    trust_remote_code can rewrite weights at init (the historical Stage 7
    packed those transformed values — a different model than GPTQ produced).
    """
    from safetensors import safe_open

    idx_file = model_path / "model.safetensors.index.json"
    if idx_file.exists():
        shard_files = sorted(set(
            json.loads(idx_file.read_text())["weight_map"].values()))
    elif (model_path / "model.safetensors").exists():
        shard_files = ["model.safetensors"]
    else:
        raise RuntimeError(f"No safetensors checkpoint found in {model_path}")

    tensors = {}
    for fname in shard_files:
        with safe_open(str(model_path / fname), framework="pt",
                       device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    print(f"[Stage 7] Loaded {len(tensors)} QDQ tensors "
          f"({len(shard_files)} shard(s))")
    return tensors


# ══════════════════════════════════════════════════════════════════════════════
# Core: pack from manifest
# ══════════════════════════════════════════════════════════════════════════════

def pack_from_manifest(model_path: Path, output_dir: Path,
                       manifest_path: Path, allow_hybrid: bool = False) -> dict:
    """Build and save the packed checkpoint. Returns the packing report.

    Raises on ANY deviation from the manifest contract (fail closed).
    """
    from quant_artifacts import (
        dequantize_artifact, load_artifact, read_quant_manifest,
    )

    manifest = read_quant_manifest(manifest_path)
    artifact_dir = Path(manifest_path).parent
    records = manifest["tensors"]

    expert_records = [r for r in records if r["kind"] == "expert_slice"]
    if expert_records and not allow_hybrid:
        raise RuntimeError(
            f"Manifest contains {len(expert_records)} GPT-OSS expert slices, "
            f"and vLLM expert packing is not implemented yet (P0.7). "
            f"Refusing to emit a checkpoint that silently leaves the dominant "
            f"expert weights in BF16. Pass --allow_hybrid to produce an "
            f"explicitly-labeled HYBRID checkpoint (experts BF16, attention "
            f"NVFP4) for debugging."
        )

    state_dict = load_qdq_state_dict(model_path)

    bytes_by = {"NVFP4_PACKED": 0, "BF16_KEPT": 0}
    counts = {"packed": 0, "expert_bf16": 0, "fallback_bf16": 0, "verified": 0}
    ignore_names = ["lm_head"]

    for r in records:
        disposition = r["disposition"]
        name, e_idx = r["name"], r["expert_index"]
        param_key = r["param"] if r["kind"] == "expert_slice" \
            else f"{r['name']}.weight"
        if param_key not in state_dict:
            raise RuntimeError(
                f"Manifest names '{param_key}' but it is missing from the "
                f"checkpoint — manifest/checkpoint mismatch (fail closed)."
            )
        qdq = state_dict[param_key]

        if disposition in ("GPTQ_NVFP4", "RTN_NVFP4"):
            art_ref = r["artifact"]
            art = load_artifact(artifact_dir, art_ref["file"], name, e_idx,
                                r["scale_block_size"], r["orig_shape"])
            # P0.6 invariant against the on-disk QDQ tensor
            target = qdq[e_idx].T if r["kind"] == "expert_slice" else qdq
            rec_w = dequantize_artifact(art).to(target.dtype)
            if not torch.equal(rec_w, target):
                diff = (rec_w.float() - target.float()).abs()
                raise RuntimeError(
                    f"P0.6 invariant violated for {name}"
                    f"{f' expert {e_idx}' if e_idx is not None else ''}: "
                    f"dequantize(artifact) != QDQ checkpoint tensor "
                    f"(max diff {diff.max().item():.3e}). Refusing to pack."
                )
            counts["verified"] += 1

            if r["kind"] == "linear":
                del state_dict[param_key]
                state_dict[f"{name}.weight"] = art.codes
                state_dict[f"{name}.weight_scale"] = art.scales
                state_dict[f"{name}.weight_scale_2"] = \
                    torch.ones(1, dtype=torch.bfloat16)
                counts["packed"] += 1
                bytes_by["NVFP4_PACKED"] += (art.codes.numel()
                                             + art.scales.numel())
            else:
                # Verified but not packable until P0.7 — stays BF16 (hybrid).
                counts["expert_bf16"] += 1

        elif disposition == "BF16_FALLBACK":
            if r["kind"] == "linear":
                ignore_names.append(name)
                counts["fallback_bf16"] += 1
            else:
                counts["expert_bf16"] += 1

        else:
            raise RuntimeError(
                f"Record {name}: unhandled disposition {disposition!r} "
                f"(fail closed)."
            )

    # Expert params (hybrid mode) go on the ignore list; count BF16 bytes.
    for pname in sorted({r["param"] for r in expert_records}):
        module = pname.rsplit(".", 1)[0] if pname.endswith(".weight") else pname
        if module not in ignore_names:
            ignore_names.append(module)
    for key, t in state_dict.items():
        if t.dtype in (torch.bfloat16, torch.float16, torch.float32):
            bytes_by["BF16_KEPT"] += t.numel() * t.element_size()

    hybrid = bool(expert_records)
    total_bytes = bytes_by["NVFP4_PACKED"] + bytes_by["BF16_KEPT"]
    report = {
        "hybrid": hybrid,
        "counts": counts,
        "bytes": bytes_by,
        "bf16_fraction": (bytes_by["BF16_KEPT"] / total_bytes
                          if total_bytes else 0.0),
        "manifest": str(manifest_path),
        "ignore": sorted(set(ignore_names)),
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    save_sharded(state_dict, output_dir)
    _copy_support_files(model_path, output_dir)
    _write_quant_config(model_path, output_dir, report["ignore"],
                        manifest["run_config"].get("nvfp4_block_size", 16))
    (output_dir / "PACKING_REPORT.json").write_text(
        json.dumps(report, indent=2))

    if hybrid:
        print("\n" + "!" * 72)
        print(f"[Stage 7] HYBRID checkpoint: {counts['expert_bf16']} expert "
              f"slices remain BF16 ({report['bf16_fraction']:.1%} of model "
              f"bytes are BF16). This is NOT a full NVFP4 model — do not "
              f"benchmark it as one. (vLLM expert packing lands with P0.7.)")
        print("!" * 72)
    print(f"[Stage 7] packed={counts['packed']} verified={counts['verified']} "
          f"fallback_bf16={counts['fallback_bf16']} "
          f"expert_bf16={counts['expert_bf16']}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Shard + save
# ══════════════════════════════════════════════════════════════════════════════

def save_sharded(state_dict: dict, output_dir: Path, max_shard_gb: float = 4.0):
    from safetensors.torch import save_file

    max_bytes = int(max_shard_gb * 1024 ** 3)
    shards, cur, cur_bytes = [], {}, 0
    for key, tensor in state_dict.items():
        sz = tensor.numel() * tensor.element_size()
        if cur_bytes + sz > max_bytes and cur:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[key] = tensor.contiguous()
        cur_bytes += sz
    if cur:
        shards.append(cur)

    if len(shards) == 1:
        path = output_dir / "model.safetensors"
        save_file(shards[0], str(path))
        print(f"[Stage 7] Saved {path.name}  "
              f"({path.stat().st_size / 1024**3:.2f} GB)")
    else:
        index = {"metadata": {"total_size": 0}, "weight_map": {}}
        for i, shard in enumerate(shards):
            fname = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, str(output_dir / fname))
            sz = (output_dir / fname).stat().st_size
            index["metadata"]["total_size"] += sz
            for k in shard:
                index["weight_map"][k] = fname
            print(f"[Stage 7] Saved {fname}  ({sz / 1024**3:.2f} GB)")
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2))


def _copy_support_files(model_path: Path, output_dir: Path):
    """Copy tokenizer/config/support files (not weights, not artifacts)."""
    for f in model_path.iterdir():
        if not f.is_file():
            continue
        if f.suffix in {".safetensors", ".bin"} or f.name.endswith(".tmp"):
            continue
        if f.name in {"config.json", "model.safetensors.index.json"}:
            continue
        dst = output_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst)


def _write_quant_config(model_path: Path, output_dir: Path,
                        ignore_names: list, block_size: int):
    config = json.loads((model_path / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "W4A16_NVFP4",
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 4, "type": "float",
                            "group_size": block_size},
                "targets": ["Linear"],
            }
        },
        "ignore": ignore_names,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[Stage 7] config.json written (quant_algo=W4A16_NVFP4, "
          f"{len(ignore_names)} ignored)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model_path", type=Path, required=True,
                   help="Stage 5 QDQ output directory")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Output directory for the packed checkpoint")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Path to quant_artifacts/manifest.json "
                        "(default: <model_path>/quant_artifacts/manifest.json)")
    p.add_argument("--allow_hybrid", action="store_true",
                   help="Permit GPT-OSS expert slices to remain BF16 "
                        "(explicitly-labeled HYBRID debug checkpoint)")
    return p.parse_args()


def main():
    args = _parse_args()
    manifest_path = (args.manifest if args.manifest is not None
                     else args.model_path / "quant_artifacts" / "manifest.json")

    print("=" * 68)
    print("Stage 7 — exact NVFP4 serialization (P0.5/P0.6)")
    print("=" * 68)
    print(f"  Model path : {args.model_path}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Manifest   : {manifest_path}")
    print(f"  Hybrid ok  : {args.allow_hybrid}")

    try:
        report = pack_from_manifest(args.model_path, args.output_dir,
                                    manifest_path, args.allow_hybrid)
    except RuntimeError as exc:
        print(f"\n[Stage 7] REFUSED: {exc}")
        sys.exit(1)

    print(f"\nStage 7 complete — checkpoint at {args.output_dir}")
    print(f"Packing report: {args.output_dir / 'PACKING_REPORT.json'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
