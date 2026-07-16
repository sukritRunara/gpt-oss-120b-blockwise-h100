#!/usr/bin/env python
"""Dequantize the official gpt-oss-20b MXFP4 checkpoint to BF16 (arm B).

Uses the pinned transformers' supported mechanism
(`Mxfp4Config(dequantize=True)`) to decode the MXFP4 expert weights and
saves a clean BF16 checkpoint named for what it is:
gpt-oss-20b-mxfp4-dequant-bf16 — a TRANSQUANTIZATION source, not the
unavailable pre-MXFP4 master (handoff §4).

Steps (handoff §11): load-dequantized → scrub quantization metadata →
save BF16 safetensors + tokenizer/config/chat template → provenance
manifest (source revision + per-file SHA-256). Idempotent: a complete
output (valid manifest, files match) is detected and skipped.

Run inside .venv-quant:
    python scripts/dequantize_gpt_oss_20b.py
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def output_complete(out_dir: Path) -> bool:
    mpath = out_dir / "PROVENANCE.json"
    if not mpath.exists():
        return False
    try:
        m = json.loads(mpath.read_text())
    except json.JSONDecodeError:
        return False
    for rel, info in m.get("files", {}).items():
        p = out_dir / rel
        if not p.exists() or p.stat().st_size != info["bytes"]:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path,
                    default=Path("/workspace/models/gpt-oss-20b-official-mxfp4"))
    ap.add_argument("--output", type=Path,
                    default=Path("/workspace/models/gpt-oss-20b-mxfp4-dequant-bf16"))
    ap.add_argument("--device_map", default="auto",
                    help="Device map for the dequantizing load (default: auto)")
    args = ap.parse_args()

    if output_complete(args.output):
        print(f"Already complete at {args.output} — nothing to do.")
        return 0

    src_prov = json.loads((args.source / "PROVENANCE.json").read_text())
    print(f"Source: {src_prov['model_id']} @ {src_prov['revision'][:12]} "
          f"({args.source})")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    print("Loading with Mxfp4Config(dequantize=True)…")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.source),
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        quantization_config=Mxfp4Config(dequantize=True),
        low_cpu_mem_usage=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params / 1e9:.3f}B parameters")

    # ── Fail-closed sanity before saving ──────────────────────────────────────
    bad_dtypes = {n: str(p.dtype) for n, p in model.named_parameters()
                  if p.dtype not in (torch.bfloat16, torch.float32)}
    if bad_dtypes:
        raise RuntimeError(f"Non-BF16/FP32 parameters after dequantize: "
                           f"{dict(list(bad_dtypes.items())[:5])}")
    leftovers = [n for n, _ in model.named_parameters()
                 if "_blocks" in n or "_scales" in n]
    if leftovers:
        raise RuntimeError(f"Packed MXFP4 tensors survived dequantize: "
                           f"{leftovers[:5]}")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(f"NaN/Inf in dequantized parameter {n}")

    # Scrub quantization metadata so the checkpoint reloads as ordinary BF16.
    if hasattr(model.config, "quantization_config"):
        try:
            del model.config.quantization_config
        except AttributeError:
            model.config.quantization_config = None
        print("  Removed quantization_config from model config")

    print(f"Saving BF16 checkpoint to {args.output} …")
    args.output.mkdir(parents=True, exist_ok=True)

    # Manual sharded save. transformers' save_pretrained runs
    # revert_weight_conversion(), which maps the dequantized expert weights
    # back to their checkpoint-format names (gate_up_proj_blocks/_scales) and
    # silently DROPS them when the reverse conversion cannot run — observed:
    # a 4.9 GB "checkpoint" missing every expert tensor. Writing the state
    # dict directly avoids all quantizer/conversion machinery.
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    n_state = sum(v.numel() for v in state_dict.values())
    if n_state < n_params:
        raise RuntimeError(
            f"state_dict holds {n_state/1e9:.3f}B params but the model has "
            f"{n_params/1e9:.3f}B — refusing to save an incomplete checkpoint.")

    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size="4GB",
    )
    for shard_file, tensor_names in split.filename_to_tensors.items():
        shard = {n: state_dict[n].contiguous() for n in tensor_names}
        save_file(shard, str(args.output / shard_file),
                  metadata={"format": "pt"})
        print(f"  wrote {shard_file} ({len(shard)} tensors)")
    if split.is_sharded:
        index = {"metadata": {"total_size": split.metadata["total_size"]},
                 "weight_map": split.tensor_to_filename}
        (args.output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2))

    # Config (scrubbed of quantization metadata) + generation config
    model.config.save_pretrained(str(args.output))
    if model.generation_config is not None:
        model.generation_config.save_pretrained(str(args.output))

    tok = AutoTokenizer.from_pretrained(str(args.source))
    tok.save_pretrained(str(args.output))
    for fname in ("chat_template.jinja", "generation_config.json"):
        src = args.source / fname
        if src.exists() and not (args.output / fname).exists():
            shutil.copy2(src, args.output / fname)

    # Belt-and-braces: the saved config.json must not claim MXFP4.
    cfg = json.loads((args.output / "config.json").read_text())
    if "quantization_config" in cfg:
        del cfg["quantization_config"]
        (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
        print("  Scrubbed quantization_config from saved config.json")

    print("Hashing outputs for provenance…")
    files = {}
    for p in sorted(args.output.rglob("*")):
        if not p.is_file() or p.name == "PROVENANCE.json":
            continue
        rel = str(p.relative_to(args.output))
        files[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    manifest = {
        "artifact": "gpt-oss-20b-mxfp4-dequant-bf16",
        "provenance": ("official MXFP4 checkpoint → exact MXFP4 decode "
                       "(transformers Mxfp4Config(dequantize=True)) → BF16. "
                       "NOT the pre-MXFP4 master (transquantization source)."),
        "source_model_id": src_prov["model_id"],
        "source_revision": src_prov["revision"],
        "source_manifest_sha": {
            rel: info["sha256"] for rel, info in src_prov["files"].items()
            if rel.endswith(".safetensors")
        },
        "transformers_version": __import__("transformers").__version__,
        "n_parameters": n_params,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    (args.output / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f["bytes"] for f in files.values())
    print(f"\nDone: {total / 1024**3:.1f} GB at {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
