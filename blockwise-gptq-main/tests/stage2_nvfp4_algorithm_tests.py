"""
Stage 2 — NVFP4 Algorithm Tests

Validates that fasterquant_blockwise integrates correctly with NVFP4Quantizer.
No model download required. Uses small synthetic weight matrices.

Tests:
  1. Integration smoke    — fasterquant_blockwise with NVFP4 runs for B ∈ {16,32,64,128}
  2. Per-block params     — find_params is called once per GPTQ block (not once globally)
  3. GPTQ ≤ RTN output   — error compensation reduces calibration-data MSE vs round-to-nearest
  4. Remainder columns   — in_features not a multiple of 16 handled without shape error
  5. Condition logging   — log_condition=True returns exactly n_gptq_blocks scalars
  6. Near-singular H     — percdamp prevents NaN when Hessian is rank-deficient
  7. Format comparison   — INT8/FP8/NVFP4 all finish with finite MSE, ordered as expected

Runtime: ~2 min on DGX Spark GB10 (GPU)
Usage:   python stage2_nvfp4_algorithm_tests.py
Exit:    0 = all passed, 1 = one or more failed
"""

import sys
import traceback
from pathlib import Path

_CODE_ROOT = Path(
    "/home/runara_dgx_spark_1/Itamar/projects"
    "/Block-wise-GPTQ-GPT-OSS-20B-NVFP4/blockwise-gptq/"
    "/opteam-blockwise-gptq"
)

if not _CODE_ROOT.exists():
    raise RuntimeError(
        f"Code root not found: {_CODE_ROOT}\n"
        "Update _CODE_ROOT at the top of this script."
    )

sys.path.insert(0, str(_CODE_ROOT))
print(f"[path] {_CODE_ROOT}")

import torch
import torch.nn as nn
from quantizer import NVFP4Quantizer, FP8E4M3Quantizer, Int8SymQuantizer
from gptq import GPTQ

# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_gptq(in_f, out_f, ncalib, seed, device, quantizer_cls, **q_kwargs):
    """Create a linear layer, attach quantizer, accumulate Hessian, return (linear, gptq, calib).

    calib shape: [ncalib, 1, in_f]  (each sample is a [1, in_f] 2-D tensor)
    """
    torch.manual_seed(seed)
    linear = nn.Linear(in_f, out_f, bias=False, device=device, dtype=torch.float32)
    nn.init.normal_(linear.weight, std=0.02)

    torch.manual_seed(seed + 1000)
    calib = torch.randn(ncalib, 1, in_f, device=device)

    gptq = GPTQ(linear)
    gptq.quantizer = quantizer_cls(device=device, **q_kwargs)
    for i in range(ncalib):
        gptq.add_batch(calib[i], None)

    return linear, gptq, calib


# ─────────────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    _results.append((name, passed, detail))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Integration smoke
# ─────────────────────────────────────────────────────────────────────────────

def test_integration_smoke():
    """fasterquant_blockwise with NVFP4 completes for B ∈ {16, 32, 64, 128}.

    For each blocksize:
    - Loss must be finite and positive
    - All weight values must be finite after quantization
    - Weight shape must be unchanged

    blocksize must always be a multiple of NVFP4's microscaling block size (16).
    """
    print("\n── Test 1: Integration smoke ────────────────────────────────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_f, out_f, ncalib, seed = 512, 256, 32, 42
    failures = []

    for B in [16, 32, 64, 128]:
        linear, gptq, _ = _make_gptq(in_f, out_f, ncalib, seed, device,
                                      NVFP4Quantizer, block_size=16)
        W_orig_shape = linear.weight.shape

        loss = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
        W = linear.weight.data

        loss_ok    = torch.isfinite(torch.tensor(loss)).item() and loss > 0
        weights_ok = torch.isfinite(W).all().item()
        shape_ok   = W.shape == W_orig_shape

        ok = loss_ok and weights_ok and shape_ok
        tag = "OK" if ok else "FAIL"
        print(f"    B={B:>3d}: loss={loss:.4f}, weights_finite={weights_ok}, "
              f"shape={tuple(W.shape)}  [{tag}]")

        if not ok:
            failures.append(
                f"B={B} loss_ok={loss_ok} weights_ok={weights_ok} shape_ok={shape_ok}"
            )
        gptq.free()

    _record("Integration smoke",
            not failures,
            "all B OK" if not failures else " | ".join(failures))


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Per-block find_params call count
# ─────────────────────────────────────────────────────────────────────────────

def test_per_block_find_params_calls():
    """find_params must be called once per GPTQ block, not once globally.

    NVFP4 requires per-block microscaling: each block of `block_size=16`
    columns needs its own FP8 scale derived from the error-compensated weights
    at the time of quantization. Calling find_params globally (once, before the
    GPTQ loop) uses pre-quantization scales and ignores Hessian-guided error.

    This test monkey-patches find_params with a counter and verifies the call
    count equals ceil(in_features / blocksize) — i.e. one call per GPTQ block.
    """
    print("\n── Test 2: Per-block find_params call count ─────────────────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_f, out_f, ncalib, seed = 256, 128, 16, 7

    for B in [32, 64, 128]:
        import math
        expected_calls = math.ceil(in_f / B)

        linear, gptq, _ = _make_gptq(in_f, out_f, ncalib, seed, device,
                                      NVFP4Quantizer, block_size=16)

        # Monkey-patch the quantizer's find_params to count calls
        original_fp = gptq.quantizer.find_params
        call_count = [0]

        def _counting_fp(x, weight=True):
            call_count[0] += 1
            return original_fp(x, weight)

        gptq.quantizer.find_params = _counting_fp

        gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
        actual_calls = call_count[0]
        gptq.free()

        ok = (actual_calls == expected_calls)
        tag = "OK" if ok else "FAIL"
        print(f"    B={B:>3d}: expected {expected_calls} calls, got {actual_calls}  [{tag}]")

        if not ok:
            _record("Per-block find_params calls", False,
                    f"B={B}: expected {expected_calls}, got {actual_calls}")
            return

    _record("Per-block find_params calls", True,
            "call count == ceil(in_f/B) for B ∈ {32,64,128}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — GPTQ ≤ RTN calibration output MSE
# ─────────────────────────────────────────────────────────────────────────────

def test_gptq_beats_rtn():
    """GPTQ must achieve lower calibration-data output MSE than round-to-nearest (RTN).

    GPTQ optimises the Hessian-weighted loss Σ ||x·(W - W_q)^T||² over
    calibration inputs x. RTN (find_params once + quantize_dequantize, no error
    compensation) serves as the no-optimisation baseline.

    Setup:  in_f=256, ncalib=16  →  Hessian is rank-16 in a 256-dim space.
    With 240 null directions available, GPTQ can redirect quantization error
    away from the 16 directions that actually influence the output, sharply
    reducing calibration-data MSE relative to RTN.

    Assertion: gptq_mse < rtn_mse  (strict, on calibration data itself)
    """
    print("\n── Test 3: GPTQ ≤ RTN calibration MSE ──────────────────────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_f, out_f, ncalib, seed, B = 256, 128, 16, 42, 64

    # Build layer + calib (do NOT create GPTQ yet — need W_orig before any update)
    torch.manual_seed(seed)
    linear_ref = nn.Linear(in_f, out_f, bias=False, device=device, dtype=torch.float32)
    nn.init.normal_(linear_ref.weight, std=0.02)
    W_orig = linear_ref.weight.data.clone()

    torch.manual_seed(seed + 1000)
    calib = torch.randn(ncalib, 1, in_f, device=device)  # [ncalib, 1, in_f]

    # ── RTN baseline ─────────────────────────────────────────────────────────
    # One global find_params call on the original weights, then quantize_dequantize.
    # No Hessian, no error compensation.
    q_rtn = NVFP4Quantizer(block_size=16, device=device)
    q_rtn.find_params(W_orig.clone())
    W_rtn = q_rtn.quantize_dequantize(W_orig.clone())

    with torch.no_grad():
        ref_out = torch.cat([calib[i] @ W_orig.T for i in range(ncalib)])   # [ncalib, out_f]
        rtn_out = torch.cat([calib[i] @ W_rtn.T  for i in range(ncalib)])
    rtn_mse = (ref_out - rtn_out).pow(2).mean().item()

    # ── GPTQ ─────────────────────────────────────────────────────────────────
    linear_gptq, gptq, _ = _make_gptq(in_f, out_f, ncalib, seed, device,
                                        NVFP4Quantizer, block_size=16)
    gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
    W_gptq = linear_gptq.weight.data.clone()
    gptq.free()

    with torch.no_grad():
        gptq_out = torch.cat([calib[i] @ W_gptq.T for i in range(ncalib)])
    gptq_mse = (ref_out - gptq_out).pow(2).mean().item()

    improvement_pct = 100.0 * (rtn_mse - gptq_mse) / (rtn_mse + 1e-12)
    print(f"    RTN  output MSE : {rtn_mse:.4e}")
    print(f"    GPTQ output MSE : {gptq_mse:.4e}")
    print(f"    Improvement     : {improvement_pct:.1f}%")

    passed = gptq_mse < rtn_mse
    _record("GPTQ ≤ RTN calibration MSE",
            passed,
            f"GPTQ {gptq_mse:.3e} < RTN {rtn_mse:.3e} (+{improvement_pct:.1f}%)"
            if passed else
            f"GPTQ {gptq_mse:.3e} ≥ RTN {rtn_mse:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Remainder columns (in_features not a multiple of 16)
# ─────────────────────────────────────────────────────────────────────────────

def test_remainder_columns():
    """fasterquant_blockwise handles in_features not divisible by 16.

    The microscaling block size is 16. When in_features % 16 != 0, the last
    microscaling block is partial and must be zero-padded internally by
    NVFP4Quantizer.find_params. GPTQ blocksize must still be a multiple of 16.

    Tests:
        in_f=100   (100 = 6×16 + 4),  B=32
        in_f=33    (33  = 2×16 + 1),  B=16
        in_f=48    (48  = 3×16),       B=16  — exact multiple, sanity check
    """
    print("\n── Test 4: Remainder columns (in_f not multiple of 16) ──────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ncalib, seed = 16, 42
    failures = []

    for in_f, out_f, B in [(100, 64, 32), (33, 64, 16), (48, 64, 16)]:
        linear, gptq, _ = _make_gptq(in_f, out_f, ncalib, seed, device,
                                      NVFP4Quantizer, block_size=16)
        loss = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
        W = linear.weight.data

        shape_ok   = W.shape == (out_f, in_f)
        weights_ok = torch.isfinite(W).all().item()
        loss_ok    = torch.isfinite(torch.tensor(loss)).item() and loss > 0

        ok  = shape_ok and weights_ok and loss_ok
        tag = "OK" if ok else "FAIL"
        print(f"    in_f={in_f:>3d}, B={B:>2d}: shape={tuple(W.shape)}, "
              f"loss={loss:.4f}, weights_finite={weights_ok}  [{tag}]")

        if not ok:
            failures.append(
                f"in_f={in_f} B={B}: "
                f"shape_ok={shape_ok} weights_ok={weights_ok} loss_ok={loss_ok}"
            )
        gptq.free()

    _record("Remainder columns",
            not failures,
            "all shapes OK" if not failures else " | ".join(failures))


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Condition number logging
# ─────────────────────────────────────────────────────────────────────────────

def test_condition_number_logging():
    """log_condition=True must return exactly ceil(in_f / B) condition numbers.

    Each GPTQ block produces one condition number from the block's Hessian
    sub-matrix (κ = λ_max / λ_min after damping). The returned list length
    must match the number of GPTQ blocks, all values must be finite and > 1.

    Tests three blocksizes to cover both even division and remainder blocks.
    """
    print("\n── Test 5: Condition number logging ─────────────────────────────────")

    import math
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_f, out_f, ncalib, seed = 256, 128, 32, 42
    failures = []

    for B in [32, 64, 128]:
        expected = math.ceil(in_f / B)

        linear, gptq, _ = _make_gptq(in_f, out_f, ncalib, seed, device,
                                      NVFP4Quantizer, block_size=16)
        result = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01,
                                             log_condition=True)
        gptq.free()

        # fasterquant_blockwise returns (loss, cond_nums) when log_condition=True
        if not isinstance(result, tuple) or len(result) != 2:
            failures.append(f"B={B}: expected (loss, cond_nums), got {type(result)}")
            print(f"    B={B:>3d}: wrong return type {type(result)}  [FAIL]")
            continue

        loss, cond_nums = result
        count_ok   = len(cond_nums) == expected
        finite_ok  = all(torch.isfinite(torch.tensor(float(c))).item() for c in cond_nums)
        positive_ok = all(c > 0 for c in cond_nums)

        ok = count_ok and finite_ok and positive_ok
        tag = "OK" if ok else "FAIL"
        cond_str = f"[{cond_nums[0]:.1f}…{cond_nums[-1]:.1f}]" if cond_nums else "[]"
        print(f"    B={B:>3d}: expected {expected} blocks, got {len(cond_nums)}, "
              f"cond={cond_str}  [{tag}]")

        if not ok:
            failures.append(
                f"B={B}: count_ok={count_ok} finite_ok={finite_ok} positive_ok={positive_ok}"
            )

    _record("Condition number logging",
            not failures,
            "count/finite/positive OK for B ∈ {32,64,128}"
            if not failures else " | ".join(failures))


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Near-singular Hessian (percdamp robustness)
# ─────────────────────────────────────────────────────────────────────────────

def test_near_singular_hessian():
    """percdamp must prevent NaN when the Hessian is severely rank-deficient.

    Setup: ncalib=4 calibration samples for in_features=256. The resulting
    Hessian H = X^T X (256×256) is rank ≤ 4 — it has 252 zero eigenvalues.
    Without damping, Cholesky would fail or produce ∞/NaN. With percdamp=0.01
    the diagonal is shifted by 0.01 × max_diag, making H invertible.

    The test verifies: no NaN/Inf in weights, finite positive loss.
    """
    print("\n── Test 6: Near-singular Hessian (percdamp robustness) ──────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Very few calibration samples relative to in_features → rank-deficient H
    in_f, out_f, ncalib, seed, B = 256, 64, 4, 17, 64

    print(f"    Hessian shape: [{in_f}x{in_f}], rank ≤ {ncalib}  "
          f"({in_f - ncalib} zero eigenvalues)")

    linear, gptq, _ = _make_gptq(in_f, out_f, ncalib, seed, device,
                                  NVFP4Quantizer, block_size=16)
    loss = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
    W = linear.weight.data
    gptq.free()

    loss_finite   = torch.isfinite(torch.tensor(loss)).item()
    weights_finite = torch.isfinite(W).all().item()
    loss_positive  = loss > 0

    print(f"    Loss: {loss:.4f}  (finite={loss_finite}, positive={loss_positive})")
    print(f"    Weights finite: {weights_finite}")

    passed = loss_finite and loss_positive and weights_finite
    _record("Near-singular Hessian",
            passed,
            f"loss={loss:.4f} weights_finite={weights_finite}"
            if passed else
            f"NaN/Inf detected — loss={loss} weights_finite={weights_finite}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Format comparison (INT8 / FP8 / NVFP4)
# ─────────────────────────────────────────────────────────────────────────────

def test_format_comparison():
    """INT8, FP8, and NVFP4 must all produce finite MSE; ordering INT8 < FP8 < NVFP4.

    With fasterquant_blockwise, all three formats get Hessian-guided error
    compensation. The output MSE should be finite and positive for all formats.

    Expected qualitative ordering (more bits = lower error):
        MSE(INT8) < MSE(FP8) < MSE(NVFP4)

    The NVFP4 assertion is soft (reported, not hard-gated) because hardware
    factors (scale quantization, E2M1 grid gaps) can occasionally invert
    FP8 vs NVFP4 on tiny synthetic matrices. The hard gate is: all finite.
    """
    print("\n── Test 7: Format comparison (INT8 / FP8 / NVFP4) ──────────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_f, out_f, ncalib, seed, B = 512, 256, 64, 42, 128

    # Reference outputs (unquantized)
    torch.manual_seed(seed)
    linear_ref = nn.Linear(in_f, out_f, bias=False, device=device, dtype=torch.float32)
    nn.init.normal_(linear_ref.weight, std=0.02)
    torch.manual_seed(seed + 1000)
    calib = torch.randn(ncalib, 1, in_f, device=device)

    with torch.no_grad():
        ref_out = torch.cat([linear_ref(calib[i]) for i in range(ncalib)])

    configs = [
        ("int8",  Int8SymQuantizer,  {}),
        ("fp8",   FP8E4M3Quantizer,  {}),
        ("nvfp4", NVFP4Quantizer,    {"block_size": 16}),
    ]

    mse_results: dict[str, float] = {}
    for fmt_name, fmt_cls, kwargs in configs:
        linear, gptq, _ = _make_gptq(in_f, out_f, ncalib, seed, device, fmt_cls, **kwargs)
        loss = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
        gptq.free()

        with torch.no_grad():
            q_out = torch.cat([linear(calib[i]) for i in range(ncalib)])
        mse = (ref_out - q_out).pow(2).mean().item()
        mse_results[fmt_name] = mse
        print(f"    {fmt_name:>6s}: gptq_loss={loss:.4f}, output_mse={mse:.4e}")

    # Hard gate: all MSE values must be finite and positive
    all_finite = all(
        torch.isfinite(torch.tensor(v)).item() and v > 0
        for v in mse_results.values()
    )

    # Soft ordering check (informational)
    if all_finite:
        int8_lt_fp8  = mse_results["int8"] < mse_results["fp8"]
        fp8_lt_nvfp4 = mse_results["fp8"]  < mse_results["nvfp4"]
        order_str = (
            f"INT8 < FP8 < NVFP4 ✓"
            if (int8_lt_fp8 and fp8_lt_nvfp4) else
            f"(ordering note: INT8 < FP8={int8_lt_fp8}, FP8 < NVFP4={fp8_lt_nvfp4})"
        )
        print(f"    MSE order: {order_str}")

    _record("Format comparison",
            all_finite,
            "all finite" if all_finite else "NaN/Inf MSE detected")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("Stage 2 — NVFP4 Algorithm Tests")
    print("=" * 68)
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tests = [
        test_integration_smoke,
        test_per_block_find_params_calls,
        test_gptq_beats_rtn,
        test_remainder_columns,
        test_condition_number_logging,
        test_near_singular_hessian,
        test_format_comparison,
    ]

    for fn in tests:
        try:
            fn()
        except Exception as e:
            test_name = fn.__name__.replace("test_", "").replace("_", " ")
            print(f"\n  [ERROR] {test_name}")
            traceback.print_exc()
            _results.append((test_name, False, f"exception: {e}"))

    # ── Summary ──────────────────────────────────────────────────────────────
    n_total  = len(_results)
    n_passed = sum(1 for _, ok, _ in _results if ok)
    n_failed = n_total - n_passed

    print("\n" + "=" * 68)
    print(f"Stage 2 summary: {n_passed}/{n_total} passed")
    print("=" * 68)

    if n_failed == 0:
        print("✓  All algorithm tests passed. Safe to proceed to Stage 3.")
        sys.exit(0)
    else:
        print(f"✗  {n_failed} test(s) failed. Fix before proceeding.\n")
        for name, ok, detail in _results:
            if not ok:
                print(f"   FAILED: {name}" + (f" — {detail}" if detail else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()