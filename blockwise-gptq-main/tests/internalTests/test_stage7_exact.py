"""Stage 7 exact-serialization tests (P0.5/P0.6).

Verifies the rewritten stage7_save_modelopt.py:
  - hard-fails without a manifest (no fail-open "pack all nn.Linear")
  - refuses GPT-OSS expert slices without --allow_hybrid, and loudly labels
    the output HYBRID when allowed
  - consumes the EXACT Stage 5 codes/scales (packed weight == artifact codes)
  - verifies every artifact against the on-disk QDQ tensor and aborts on
    tampering (P0.6 invariant with teeth)
  - packs a dense (no-expert) model fully, with correct key layout/dtypes
    and a coherent quantization_config ignore list

Run (from anywhere, in .venv-quant):
    pytest -q tests/internalTests/test_stage7_exact.py
    python  tests/internalTests/test_stage7_exact.py
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
_TESTS_ROOT = Path(__file__).resolve().parents[1]
_CODE_ROOT = _TESTS_ROOT.parent / "opteam-blockwise-gptq"
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import apply as apply_mod                                # noqa: E402
from quant_artifacts import write_quant_manifest         # noqa: E402
from test_hessian_grouped_collection import (            # noqa: E402
    _tiny_gpt_oss, _fake_calibration, N_LAYERS, SEQLEN,
)

# Import stage7 as a module (it lives outside a package)
_spec = importlib.util.spec_from_file_location(
    "stage7", _TESTS_ROOT / "stage7_save_modelopt.py")
stage7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage7)

torch.manual_seed(0)


def _quantize_and_save(model, workdir, threshold=None):
    """Quantize with artifacts, save_pretrained, write manifest. Returns paths."""
    workdir = Path(workdir)
    model_dir = workdir / "qdq"
    art_dir = model_dir / "quant_artifacts"

    calib = _fake_calibration()
    orig_loader = apply_mod.get_calibration_data
    apply_mod.get_calibration_data = lambda *a, **k: calib
    try:
        _, _, _, records = apply_mod.gptq_quantize_model(
            model, "tiny", quant_format="nvfp4", dataset="synthetic",
            nsamples=len(calib), seqlen=SEQLEN, blocksize=32, percdamp=0.01,
            seed=0, device="cpu", mode="blockwise", parallel_hessian=True,
            mixed_precision_threshold=threshold,
            hessian_cache_dir=str(workdir / "hcache"),
            hessian_layer_group_size=N_LAYERS,
            artifact_dir=str(art_dir),
        )
    finally:
        apply_mod.get_calibration_data = orig_loader

    model.save_pretrained(str(model_dir))
    covered = {r["param"] for r in records}
    excluded = [{"param": n, "reason": "not eligible"}
                for n, _ in model.named_parameters() if n not in covered]
    manifest_path = write_quant_manifest(
        art_dir, records,
        {"quant_format": "nvfp4", "nvfp4_block_size": 16}, excluded)
    return model_dir, Path(manifest_path), records


def _tiny_llama():
    """Dense control model (no experts) — full-NVFP4 packing must succeed."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        max_position_embeddings=128,
    )
    torch.manual_seed(11)
    m = LlamaForCausalLM(cfg).float()
    m.eval()
    return m


# Shared GPT-OSS fixture (quantization is the slow part)
_GPTOSS = {}


def _gptoss_fixture():
    if not _GPTOSS:
        tmp = tempfile.TemporaryDirectory()
        model_dir, manifest, records = _quantize_and_save(_tiny_gpt_oss(), tmp.name)
        _GPTOSS.update(tmp=tmp, model_dir=model_dir,
                       manifest=manifest, records=records)
    return _GPTOSS


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_refuses_without_manifest():
    f = _gptoss_fixture()
    with tempfile.TemporaryDirectory() as out:
        with pytest.raises(RuntimeError, match="refuses"):
            stage7.pack_from_manifest(
                f["model_dir"], Path(out),
                f["model_dir"] / "quant_artifacts" / "nope.json")


def test_full_pack_includes_experts():
    """No fallbacks → experts pack to the vLLM FusedMoE layout (contract §4),
    linears carry ModelOpt global scales, output is NOT hybrid."""
    f = _gptoss_fixture()
    from quant_artifacts import load_artifact
    art_dir = f["manifest"].parent

    with tempfile.TemporaryDirectory() as out:
        out = Path(out)
        report = stage7.pack_from_manifest(f["model_dir"], out, f["manifest"],
                                           allow_hybrid=False)
        assert report["hybrid"] is False
        assert report["counts"]["verified"] == len(f["records"])
        assert report["counts"]["experts_packed"] > 0

        packed = stage7.load_qdq_state_dict(out)

        # Linears: exact codes/scales + fp32 ModelOpt global scale
        for r in f["records"]:
            if r["kind"] != "linear":
                continue
            art = load_artifact(art_dir, r["artifact"]["file"], r["name"],
                                None, r["scale_block_size"], r["orig_shape"])
            w = packed[f"{r['name']}.weight"]
            s = packed[f"{r['name']}.weight_scale"]
            s2 = packed[f"{r['name']}.weight_scale_2"]
            assert w.dtype == torch.uint8 and torch.equal(w, art.codes)
            assert s.dtype == torch.float8_e4m3fn
            assert torch.equal(s.to(torch.float32),
                               art.scales.to(torch.float32))
            assert s2.dtype == torch.float32
            assert torch.equal(s2, art.global_scale.to(torch.float32))

        # q/k/v share one global scale (vLLM fuses them; max() is applied
        # without rescaling fp8 groups — contract §3)
        for li in range(N_LAYERS):
            prefix = f"model.layers.{li}.self_attn"
            s2s = [packed[f"{prefix}.{p}.weight_scale_2"].item()
                   for p in ("q_proj", "k_proj", "v_proj")]
            assert s2s[0] == s2s[1] == s2s[2], f"layer {li} qkv scale_2 differ"

        # Experts: FusedMoE layout, HF orientation, transposed exact codes
        by_layer = {}
        for r in f["records"]:
            if r["kind"] == "expert_slice":
                by_layer.setdefault(r["layer_index"], {}).setdefault(
                    r["projection"], {})[r["expert_index"]] = r
        for li, sides in by_layer.items():
            prefix = f"model.layers.{li}.mlp.experts"
            assert f"{prefix}.gate_up_proj" not in packed
            assert f"{prefix}.down_proj" not in packed
            n_exp = len(sides["gate_up"])

            w13 = packed[f"{prefix}.w13_weight"]
            s13 = packed[f"{prefix}.w13_weight_scale"]
            g13 = packed[f"{prefix}.w13_weight_scale_2"]
            i13 = packed[f"{prefix}.w13_input_scale"]
            assert w13.dtype == torch.uint8 and w13.shape[0] == n_exp
            assert s13.dtype == torch.float8_e4m3fn
            assert g13.dtype == torch.float32 and g13.shape == (n_exp, 2)
            assert torch.equal(g13[:, 0], g13[:, 1])
            assert torch.all(i13 == 1.0)

            for e in range(n_exp):
                r = sides["gate_up"][e]
                art = load_artifact(art_dir, r["artifact"]["file"], r["name"],
                                    e, r["scale_block_size"], r["orig_shape"])
                assert torch.equal(w13[e], art.codes.T)
                assert torch.equal(s13[e].to(torch.float32),
                                   art.scales.T.to(torch.float32))
                assert g13[e, 0] == art.global_scale.item()

            w2 = packed[f"{prefix}.w2_weight"]
            g2 = packed[f"{prefix}.w2_weight_scale_2"]
            assert w2.dtype == torch.uint8 and w2.shape[0] == n_exp
            assert g2.shape == (n_exp,)
            for e in range(n_exp):
                r = sides["down"][e]
                art = load_artifact(art_dir, r["artifact"]["file"], r["name"],
                                    e, r["scale_block_size"], r["orig_shape"])
                assert torch.equal(w2[e], art.codes.T)

        cfg = json.loads((out / "config.json").read_text())
        qc = cfg["quantization_config"]
        assert qc["quant_algo"] == "W4A16_NVFP4"
        assert qc["group_size"] == 16            # flat, where vLLM reads it
        assert qc["ignore"] == ["lm_head"]       # experts are packed, not ignored
        assert (out / "PACKING_REPORT.json").exists()


def test_expert_fallback_refused_without_hybrid_flag():
    """Force expert BF16 fallbacks via a tiny threshold: packing must refuse
    without --allow_hybrid and label HYBRID with it."""
    with tempfile.TemporaryDirectory() as work:
        model_dir, manifest, records = _quantize_and_save(
            _tiny_gpt_oss(), work, threshold=1e-9)
        assert any(r["kind"] == "expert_slice"
                   and r["disposition"] == "BF16_FALLBACK" for r in records)

        out = Path(work) / "packed"
        with pytest.raises(RuntimeError, match="not NVFP4"):
            stage7.pack_from_manifest(model_dir, out, manifest,
                                      allow_hybrid=False)

        report = stage7.pack_from_manifest(model_dir, out, manifest,
                                           allow_hybrid=True)
        assert report["hybrid"] is True
        cfg = json.loads((out / "config.json").read_text())
        assert any(".mlp.experts" in n
                   for n in cfg["quantization_config"]["ignore"])


def test_tampered_artifact_aborts_pack():
    """Corrupt one artifact shard → the P0.6 verification must refuse."""
    f = _gptoss_fixture()
    from safetensors import safe_open
    from safetensors.torch import save_file

    art_dir = f["manifest"].parent
    shard = art_dir / "artifacts_layer_00.safetensors"
    tensors = {}
    with safe_open(str(shard), framework="pt", device="cpu") as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k)
    key = next(k for k in tensors if k.endswith(".codes"))
    tensors[key] = tensors[key].clone()
    tensors[key][0, 0] ^= 0x11        # flip nibbles in one byte
    backup = shard.read_bytes()
    try:
        save_file(tensors, str(shard))
        with tempfile.TemporaryDirectory() as out:
            with pytest.raises(RuntimeError, match="P0.6 invariant"):
                stage7.pack_from_manifest(f["model_dir"], Path(out),
                                          f["manifest"], allow_hybrid=True)
    finally:
        shard.write_bytes(backup)     # restore for other tests


def test_dense_model_full_pack():
    """No experts → pack succeeds WITHOUT allow_hybrid; nothing left BF16
    except embeddings/norms/lm_head; report says hybrid=False."""
    with tempfile.TemporaryDirectory() as work:
        model_dir, manifest, records = _quantize_and_save(_tiny_llama(), work)
        assert all(r["kind"] == "linear" for r in records)

        out = Path(work) / "packed"
        report = stage7.pack_from_manifest(model_dir, out, manifest,
                                           allow_hybrid=False)
        assert report["hybrid"] is False
        assert report["counts"]["packed"] == len(records)
        assert report["counts"]["expert_bf16"] == 0

        packed = stage7.load_qdq_state_dict(out)
        for r in records:
            assert packed[f"{r['name']}.weight"].dtype == torch.uint8
            assert (packed[f"{r['name']}.weight"].shape
                    == (r["orig_shape"][0], r["orig_shape"][1] // 2))


# ── Script entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
