"""End-to-end P0.5/P0.6 test: capture → manifest → exact reload.

Runs the real gptq_quantize_model() on a tiny GptOssForCausalLM with
artifact_dir set, then verifies the full manifest contract:

  - every eligible tensor (12 attn linears + 2·N_experts expert slices per
    MoE layer) has exactly one record with all REQUIRED_TENSOR_FIELDS
  - every GPTQ_NVFP4 / RTN_NVFP4 record has a loadable artifact whose
    dequantization reproduces the in-model QDQ weight bit-for-bit —
    including batched expert slices (via the transpose orientation)
  - records ∪ excluded covers every named parameter of the model exactly
  - write_quant_manifest / read_quant_manifest round-trip, and validation
    hard-errors on missing fields or artifact-less NVFP4 dispositions
  - BF16_FALLBACK records appear when the mixed-precision threshold trips,
    with no artifact and the weight left at its original value

Run (from anywhere, in .venv-quant):
    pytest -q tests/internalTests/test_manifest_e2e.py
    python  tests/internalTests/test_manifest_e2e.py
"""

import copy
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
_CODE_ROOT = Path(__file__).resolve().parents[2] / "opteam-blockwise-gptq"
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import apply as apply_mod                                # noqa: E402
from quant_artifacts import (                            # noqa: E402
    MANIFEST_NAME, REQUIRED_TENSOR_FIELDS,
    dequantize_artifact, load_artifact,
    read_quant_manifest, write_quant_manifest,
)
from test_hessian_grouped_collection import (            # noqa: E402
    _tiny_gpt_oss, _fake_calibration, N_LAYERS, N_EXPERTS, SEQLEN, N_SAMPLES,
)

torch.manual_seed(0)


def _quantize_with_artifacts(model, cache_root, artifact_dir, threshold=None):
    calib = _fake_calibration()
    orig_loader = apply_mod.get_calibration_data
    apply_mod.get_calibration_data = lambda *a, **k: calib
    try:
        return apply_mod.gptq_quantize_model(
            model, "tiny-gpt-oss",
            quant_format="nvfp4", dataset="synthetic",
            nsamples=len(calib), seqlen=SEQLEN,
            blocksize=32, percdamp=0.01, seed=0, device="cpu",
            mode="blockwise", parallel_hessian=True,
            mixed_precision_threshold=threshold,
            hessian_cache_dir=str(cache_root),
            hessian_layer_group_size=N_LAYERS,
            artifact_dir=str(artifact_dir),
        )
    finally:
        apply_mod.get_calibration_data = orig_loader


def _run_once():
    """Quantize a tiny model once; return (model, records, artifact_dir ctx)."""
    model = _tiny_gpt_oss()
    tmp = tempfile.TemporaryDirectory()
    art_dir = Path(tmp.name) / "quant_artifacts"
    _, _, _, records = _quantize_with_artifacts(
        model, Path(tmp.name) / "hcache", art_dir)
    return model, records, art_dir, tmp


# Share one quantization run across assertions (CPU GPTQ is the slow part)
_SHARED = {}


def _shared():
    if not _SHARED:
        model, records, art_dir, tmp = _run_once()
        _SHARED.update(model=model, records=records, art_dir=art_dir, tmp=tmp)
    return _SHARED


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_every_eligible_tensor_has_a_record():
    s = _shared()
    records = s["records"]
    # Per layer: 4 attention linears + 2 sides × N_EXPERTS expert slices.
    expected = N_LAYERS * (4 + 2 * N_EXPERTS)
    assert len(records) == expected, f"{len(records)} records != {expected}"

    for r in records:
        for f in REQUIRED_TENSOR_FIELDS:
            assert f in r, f"record {r.get('name')} missing {f}"
    # No duplicates
    keys = {(r["name"], r["expert_index"]) for r in records}
    assert len(keys) == len(records)
    # All quantized in this run (no threshold)
    assert all(r["disposition"] == "GPTQ_NVFP4" for r in records)
    assert all(r["hessian_nsamples"] > 0 for r in records)


def test_artifacts_reproduce_model_weights_bitexact():
    """The invariant, end-to-end: load each artifact from its shard and
    compare against the in-model QDQ weight — linears directly, expert
    slices through the transpose orientation."""
    s = _shared()
    model, records, art_dir = s["model"], s["records"], s["art_dir"]
    params = dict(model.named_parameters())

    for r in records:
        art = load_artifact(art_dir, r["artifact"]["file"], r["name"],
                            r["expert_index"], r["scale_block_size"],
                            r["orig_shape"])
        rec_w = dequantize_artifact(art)
        if r["kind"] == "linear":
            target = params[r["param"]].data
            assert torch.equal(rec_w.to(target.dtype), target), r["name"]
        else:
            batched = params[r["param"]].data[r["expert_index"]]
            assert torch.equal(rec_w.to(batched.dtype), batched.T), \
                f"{r['name']} expert {r['expert_index']}"


def test_manifest_roundtrip_and_coverage():
    s = _shared()
    model, records, art_dir = s["model"], s["records"], s["art_dir"]

    covered = {r["param"] for r in records}
    excluded = [{"param": n, "reason": "not eligible"}
                for n, _ in model.named_parameters() if n not in covered]
    path = write_quant_manifest(art_dir, records,
                                {"quant_format": "nvfp4"}, excluded)
    m = read_quant_manifest(path)

    assert m["counts"]["GPTQ_NVFP4"] == len(records)
    # records ∪ excluded == all named parameters, no overlap
    all_params = {n for n, _ in model.named_parameters()}
    manifest_params = ({r["param"] for r in m["tensors"]}
                       | {e["param"] for e in m["excluded"]})
    assert manifest_params == all_params
    assert not ({r["param"] for r in m["tensors"]}
                & {e["param"] for e in m["excluded"]})


def test_manifest_validation_fails_closed():
    s = _shared()
    records, art_dir = s["records"], s["art_dir"]

    # Missing field → write refuses
    bad = [dict(records[0])]
    del bad[0]["hessian_nsamples"]
    with pytest.raises(ValueError, match="missing fields"):
        write_quant_manifest(art_dir, bad, {}, [])

    # NVFP4 disposition without artifact → write refuses
    bad = [dict(records[0])]
    bad[0]["artifact"] = None
    with pytest.raises(ValueError, match="requires an artifact"):
        write_quant_manifest(art_dir, bad, {}, [])

    # Reader: missing manifest → hard error mentioning fail-closed policy
    with pytest.raises(RuntimeError, match="refuses"):
        read_quant_manifest(Path(art_dir) / "does_not_exist.json")

    # Reader: tampered record → hard error
    good_path = write_quant_manifest(art_dir, s["records"],
                                     {"quant_format": "nvfp4"}, [])
    m = json.loads(Path(good_path).read_text())
    del m["tensors"][0]["disposition"]
    tampered = Path(art_dir) / "tampered.json"
    tampered.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="missing required fields"):
        read_quant_manifest(tampered)


def test_bf16_fallback_recorded_without_artifact():
    """A tiny threshold forces fallbacks; those records must carry
    BF16_FALLBACK, a reason, no artifact — and the weight must be untouched."""
    model = _tiny_gpt_oss()
    original = {n: p.detach().clone() for n, p in model.named_parameters()}

    with tempfile.TemporaryDirectory() as d:
        art_dir = Path(d) / "quant_artifacts"
        _, _, _, records = _quantize_with_artifacts(
            model, Path(d) / "hcache", art_dir, threshold=1e-9)

    fallbacks = [r for r in records if r["disposition"] == "BF16_FALLBACK"]
    assert fallbacks, "threshold=1e-9 produced no fallbacks?"
    for r in fallbacks:
        assert r["artifact"] is None
        assert r["reason"] and "threshold" in r["reason"]
    # Fallback linears keep their original weights
    params = dict(model.named_parameters())
    for r in fallbacks:
        if r["kind"] == "linear":
            assert torch.equal(params[r["param"]].data, original[r["param"]]), \
                f"{r['name']} was modified despite BF16_FALLBACK"


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
