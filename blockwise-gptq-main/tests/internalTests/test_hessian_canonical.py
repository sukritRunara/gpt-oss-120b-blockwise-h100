"""Canonical Hessian accumulation tests (P0.3).

One accumulation convention for the whole pipeline:

    H = (2/N) · Σ_rows x xᵀ     N = flattened activation rows

implemented once in gptq.accumulate_hessian() and used by BOTH
GPTQ.add_batch (dense linears) and expert_dispatch._GptqH.add_batch
(GPT-OSS experts, parallel mode).

Covers the handoff §12 / P0.3 battery:
  - chunked accumulation == one-shot (equal chunks)
  - UNEQUAL chunk sizes == one-shot  (the case the old _GptqH got wrong)
  - 2D and 3D activation inputs
  - GPTQ.add_batch ≡ _GptqH.add_batch on the same activation stream
  - direct-formula check: H == (2/N)·XᵀX (no stray n / 1/n / sqrt(n) factors)
  - save→load round trip preserves (H, nsamples) and the quantized output
  - zero-row batches are a no-op

Run (from anywhere, in .venv-quant):
    pytest -q tests/internalTests/test_hessian_canonical.py
    python  tests/internalTests/test_hessian_canonical.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────
_CODE_ROOT = Path(__file__).resolve().parents[2] / "opteam-blockwise-gptq"
sys.path.insert(0, str(_CODE_ROOT))

from gptq import GPTQ, accumulate_hessian          # noqa: E402
from expert_dispatch import _GptqH                 # noqa: E402

torch.manual_seed(0)

IN_FEATURES = 96


def _direct_hessian(X: torch.Tensor) -> torch.Tensor:
    """Reference: H = (2/N)·XᵀX computed in one shot from the full stream."""
    X = X.reshape(-1, X.shape[-1]).float()
    return (2.0 / X.shape[0]) * (X.t() @ X)


def _gptq_for(in_features=IN_FEATURES) -> GPTQ:
    layer = nn.Linear(in_features, 32, bias=False)
    return GPTQ(layer)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_direct_formula_no_stray_factors():
    """H must equal (2/N)·XᵀX exactly — catches extra n, 1/n, sqrt(n) factors."""
    X = torch.randn(512, IN_FEATURES)
    g = _gptq_for()
    g.add_batch(X, None)
    torch.testing.assert_close(g.H, _direct_hessian(X), rtol=1e-5, atol=1e-5)
    assert g.nsamples == 512


def test_chunked_equals_oneshot_equal_chunks():
    X = torch.randn(400, IN_FEATURES)
    g_full, g_chunk = _gptq_for(), _gptq_for()
    g_full.add_batch(X, None)
    for c in X.chunk(8):
        g_chunk.add_batch(c, None)
    torch.testing.assert_close(g_chunk.H, g_full.H, rtol=1e-5, atol=1e-5)
    assert g_chunk.nsamples == g_full.nsamples == 400


def test_chunked_equals_oneshot_unequal_chunks():
    """Wildly unequal chunk sizes — mimics per-expert token counts that vary
    per calibration sample. This is the case the old _GptqH formula failed."""
    sizes = [1, 7, 128, 3, 61, 200, 2, 98]           # sums to 500
    X = torch.randn(sum(sizes), IN_FEATURES)
    g_full, g_chunk = _gptq_for(), _gptq_for()
    g_full.add_batch(X, None)
    off = 0
    for s in sizes:
        g_chunk.add_batch(X[off:off + s], None)
        off += s
    torch.testing.assert_close(g_chunk.H, g_full.H, rtol=1e-5, atol=1e-5)


def test_3d_input_flattening():
    X3 = torch.randn(8, 32, IN_FEATURES)             # [batch, seq, features]
    g3, g2 = _gptq_for(), _gptq_for()
    g3.add_batch(X3, None)
    g2.add_batch(X3.reshape(-1, IN_FEATURES), None)
    torch.testing.assert_close(g3.H, g2.H, rtol=1e-6, atol=1e-6)
    assert g3.nsamples == 8 * 32


def test_gptqh_equals_gptq_same_stream():
    """The two accumulator classes must be numerically identical on the same
    unequal-chunk stream — the exact regression the old code had."""
    sizes = [5, 90, 1, 44, 260, 12]
    X = torch.randn(sum(sizes), IN_FEATURES)

    g = _gptq_for()
    h = _GptqH(IN_FEATURES)
    off = 0
    for s in sizes:
        g.add_batch(X[off:off + s], None)
        h.add_batch(X[off:off + s], None)
        off += s

    assert h.nsamples == g.nsamples
    torch.testing.assert_close(h.H, g.H, rtol=1e-5, atol=1e-5)
    # And both equal the direct formula
    torch.testing.assert_close(h.H, _direct_hessian(X), rtol=1e-5, atol=1e-5)


def test_gptqh_transfer_into_gptq_quantizes_identically():
    """Cached-Hessian pathway: (H, nsamples) collected by _GptqH and
    transplanted into a GPTQ instance must produce the same quantized weights
    as native GPTQ.add_batch collection."""
    sys.path.insert(0, str(_CODE_ROOT))
    from quantizer import NVFP4Quantizer

    torch.manual_seed(3)
    W = torch.randn(32, IN_FEATURES)
    X = torch.randn(300, IN_FEATURES)

    def _fresh(weight):
        layer = nn.Linear(IN_FEATURES, 32, bias=False)
        layer.weight.data.copy_(weight)
        g = GPTQ(layer)
        g.quantizer = NVFP4Quantizer(block_size=16, device="cpu")
        return g

    # Path A: native GPTQ accumulation
    g_a = _fresh(W)
    for c in X.chunk(5):
        g_a.add_batch(c, None)
    loss_a = g_a.fasterquant_blockwise(blocksize=32, percdamp=0.01)
    W_a = g_a.layer.weight.data.clone()

    # Path B: _GptqH accumulation → transplant
    h = _GptqH(IN_FEATURES)
    for c in X.chunk(5):
        h.add_batch(c, None)
    g_b = _fresh(W)
    g_b.H, g_b.nsamples = h.H.clone(), h.nsamples
    loss_b = g_b.fasterquant_blockwise(blocksize=32, percdamp=0.01)
    W_b = g_b.layer.weight.data.clone()

    torch.testing.assert_close(W_a, W_b, rtol=1e-6, atol=1e-6)
    assert abs(loss_a - loss_b) < 1e-4


def test_save_load_roundtrip_preserves_quantization():
    """Serialize (H, nsamples) the way the cache does (torch.save/load) and
    verify the reloaded state quantizes to the same weights."""
    import tempfile, os
    from quantizer import NVFP4Quantizer

    torch.manual_seed(4)
    W = torch.randn(32, IN_FEATURES)
    X = torch.randn(256, IN_FEATURES)

    h = _GptqH(IN_FEATURES)
    h.add_batch(X, None)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.pt")
        torch.save({"H": h.H.cpu(), "nsamples": h.nsamples}, p)
        payload = torch.load(p, map_location="cpu", weights_only=False)

    def _quant_with(H, n):
        layer = nn.Linear(IN_FEATURES, 32, bias=False)
        layer.weight.data.copy_(W)
        g = GPTQ(layer)
        g.quantizer = NVFP4Quantizer(block_size=16, device="cpu")
        g.H, g.nsamples = H.clone(), n
        g.fasterquant_blockwise(blocksize=32, percdamp=0.01)
        return g.layer.weight.data.clone()

    torch.testing.assert_close(
        _quant_with(h.H, h.nsamples),
        _quant_with(payload["H"], payload["nsamples"]),
        rtol=0, atol=0,
    )


def test_zero_row_batch_is_noop():
    X = torch.randn(64, IN_FEATURES)
    g_a, g_b = _gptq_for(), _gptq_for()
    g_a.add_batch(X, None)
    g_b.add_batch(X, None)
    g_b.add_batch(torch.empty(0, IN_FEATURES), None)
    assert g_b.nsamples == g_a.nsamples
    torch.testing.assert_close(g_b.H, g_a.H, rtol=0, atol=0)


def test_accumulate_hessian_pure_function_contract():
    """accumulate_hessian returns updated (H, n) without mutating callers'
    bookkeeping incorrectly across mixed 2D/3D streams."""
    H = torch.zeros(IN_FEATURES, IN_FEATURES)
    n = 0
    X1 = torch.randn(2, 10, IN_FEATURES)   # 20 rows, 3D
    X2 = torch.randn(30, IN_FEATURES)      # 30 rows, 2D
    H, n = accumulate_hessian(H, n, X1)
    H, n = accumulate_hessian(H, n, X2)
    assert n == 50
    ref = _direct_hessian(torch.cat([X1.reshape(-1, IN_FEATURES), X2]))
    torch.testing.assert_close(H, ref, rtol=1e-5, atol=1e-5)


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
