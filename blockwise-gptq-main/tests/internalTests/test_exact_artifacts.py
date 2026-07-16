"""Exact NVFP4 artifact capture tests (P0.6 core).

The invariant under test (handoff §P0.6):

    Stage 5 QDQ weight == dequantize(captured codes/scales)   (bit-exact)

plus a regression demonstration of WHY exact preservation is mandatory:
re-deriving scales from QDQ values (the old Stage 7 approach) reconstructs a
DIFFERENT model whenever a microblock's max never hit the ±6.0 grid point.

Run (from anywhere, in .venv-quant):
    pytest -q tests/internalTests/test_exact_artifacts.py
    python  tests/internalTests/test_exact_artifacts.py
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────
_CODE_ROOT = Path(__file__).resolve().parents[2] / "opteam-blockwise-gptq"
sys.path.insert(0, str(_CODE_ROOT))

from gptq import GPTQ                                   # noqa: E402
from quantizer import NVFP4Quantizer                    # noqa: E402
from quant_artifacts import (                           # noqa: E402
    QuantizedTensorArtifact, dequantize_artifact,
    pack_nibbles, unpack_nibbles, nibbles_to_values,
)

torch.manual_seed(0)


def _gptq_capture_roundtrip(out_f, in_f, blocksize, seed=0):
    """Run fasterquant_blockwise with capture; return (QDQ weight fp32, artifact)."""
    torch.manual_seed(seed)
    layer = nn.Linear(in_f, out_f, bias=False)
    W0 = torch.randn(out_f, in_f)
    layer.weight.data.copy_(W0)

    g = GPTQ(layer)
    g.quantizer = NVFP4Quantizer(block_size=16, device="cpu")
    X = torch.randn(256, in_f)
    g.add_batch(X, None)

    g.quantizer.begin_capture(out_f, in_f)
    g.fasterquant_blockwise(blocksize=blocksize, percdamp=0.01)
    art = g.quantizer.end_capture()
    return layer.weight.data.float(), art


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_pack_unpack_roundtrip():
    nib = torch.randint(0, 16, (8, 64), dtype=torch.uint8)
    assert torch.equal(unpack_nibbles(pack_nibbles(nib)), nib)


def test_nibble_values_cover_grid():
    nib = torch.arange(16, dtype=torch.uint8).unsqueeze(0)
    vals = nibbles_to_values(nib)[0]
    expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    torch.testing.assert_close(vals, expected, rtol=0, atol=0)


def test_gptq_capture_bitexact_roundtrip():
    """dequantize(artifact) must reproduce the GPTQ QDQ weight bit-for-bit."""
    W_qdq, art = _gptq_capture_roundtrip(out_f=32, in_f=128, blocksize=32)
    W_rec = dequantize_artifact(art)
    assert torch.equal(W_rec, W_qdq), \
        f"max diff {(W_rec - W_qdq).abs().max().item():.3e}"


def test_gptq_capture_partial_final_block():
    """in_features not a multiple of the GPTQ blocksize (2880 % 128 != 0 case):
    the final partial GPTQ block must still capture exactly."""
    W_qdq, art = _gptq_capture_roundtrip(out_f=16, in_f=160, blocksize=128)
    assert art.shape == (16, 160)
    assert torch.equal(dequantize_artifact(art), W_qdq)


def test_rtn_capture_bitexact_roundtrip():
    """RTN path: single full-width find_params + quantize_dequantize."""
    torch.manual_seed(3)
    W = torch.randn(24, 96)
    q = NVFP4Quantizer(block_size=16, device="cpu")
    q.begin_capture(24, 96)
    q.find_params(W)
    W_qdq = q.quantize_dequantize(W)
    art = q.end_capture()
    assert torch.equal(dequantize_artifact(art), W_qdq)


def test_bf16_cast_stays_bitexact():
    """The model stores QDQ weights in bf16; artifact dequant + same cast must
    match bit-for-bit (this is the form Stage 7 verifies)."""
    W_qdq, art = _gptq_capture_roundtrip(out_f=32, in_f=128, blocksize=64, seed=7)
    assert torch.equal(dequantize_artifact(art).to(torch.bfloat16),
                       W_qdq.to(torch.bfloat16))


def test_requantization_of_non_qdq_source_drifts():
    """P0.6 regression demo, the failure mode the old Stage 7 actually had:
    it re-quantized weights loaded from the RAW on-disk safetensors
    (`_load_all_safetensors`), not the QDQ tensors GPTQ produced. Any
    difference between the two bases (model-init transforms, or simply
    packing a non-QDQ tensor) yields a packed model that is NOT the model
    GPTQ optimized. Simulate with a slightly transformed source.
    """
    W_qdq, art = _gptq_capture_roundtrip(out_f=16, in_f=64, blocksize=32, seed=5)

    # "Raw on-disk" weights that differ from the QDQ basis by a transform
    # (as with trust_remote_code weight rewrites). 20% exceeds both the FP8
    # scale resolution and the E2M1 rounding basin, so the repack must differ.
    W_raw = W_qdq * 1.2

    # Old approach: re-derive scales/codes from that source
    q2 = NVFP4Quantizer(block_size=16, device="cpu")
    q2.find_params(W_raw)
    W_repacked = q2.quantize_dequantize(W_raw)

    assert not torch.equal(W_repacked, W_qdq), \
        "re-quantizing a non-QDQ basis coincidentally matched QDQ"
    # The exact artifact, by construction, reproduces the optimized model
    assert torch.equal(dequantize_artifact(art), W_qdq)


def test_equivalence_check_has_teeth():
    """The Stage 7 invariant check must catch a corrupted artifact: perturb
    one scale and the dequantized tensor must no longer match QDQ."""
    W_qdq, art = _gptq_capture_roundtrip(out_f=8, in_f=64, blocksize=32, seed=6)
    scales = art.scales.clone()
    s = scales[0, 0].to(torch.float32)
    bumped = (s * 2.0).to(torch.float8_e4m3fn)
    scales[0, 0] = bumped
    tampered = QuantizedTensorArtifact(
        codes=art.codes, scales=scales,
        block_size=art.block_size, shape=art.shape,
    )
    assert not torch.equal(dequantize_artifact(tampered), W_qdq)


def test_global_scale_roundtrip_bitexact():
    """D-010: with the ModelOpt per-tensor global scale set, the round trip
    must stay bit-exact and the artifact must carry the global scale."""
    torch.manual_seed(9)
    layer = nn.Linear(128, 32, bias=False)
    W0 = torch.randn(32, 128) * 0.02          # small weights → fp8-subnormal
    layer.weight.data.copy_(W0)               # territory without normalization

    g = GPTQ(layer)
    g.quantizer = NVFP4Quantizer(block_size=16, device="cpu")
    g.quantizer.set_global_scale_from(layer.weight.data)
    s2 = g.quantizer.global_scale
    assert s2 is not None and abs(s2 - W0.abs().amax().item() / 2688.0) < 1e-12

    g.add_batch(torch.randn(256, 128), None)
    g.quantizer.begin_capture(32, 128)
    g.fasterquant_blockwise(blocksize=32, percdamp=0.01)
    art = g.quantizer.end_capture()

    assert art.global_scale.item() == pytest.approx(s2)
    assert torch.equal(dequantize_artifact(art), layer.weight.data.float())
    # Normalized fp8 scales should sit high in the fp8 range, not subnormal
    assert art.scales.to(torch.float32).max().item() > 64.0


def test_capture_incomplete_coverage_raises():
    q = NVFP4Quantizer(block_size=16, device="cpu")
    q.begin_capture(4, 64)
    W = torch.randn(4, 32)
    q.find_params(W)
    q.quantize_dequantize(W, col_start=0)     # only half the columns
    with pytest.raises(RuntimeError, match="incomplete"):
        q.end_capture()


def test_capture_rejects_unaligned_width():
    q = NVFP4Quantizer(block_size=16, device="cpu")
    with pytest.raises(ValueError, match="block_size"):
        q.begin_capture(4, 60)


def test_abort_capture_resets_state():
    q = NVFP4Quantizer(block_size=16, device="cpu")
    q.begin_capture(4, 32)
    q.abort_capture()
    W = torch.randn(4, 32)
    q.find_params(W)
    q.quantize_dequantize(W)                  # must not try to capture
    with pytest.raises(RuntimeError):
        q.end_capture()                        # nothing active


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
