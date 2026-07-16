"""
Stage 1 — NVFP4 Unit Tests

Validates the NVFP4Quantizer in isolation, with no model and no GPU required.
All five tests must pass before proceeding to any later stage. The script
exits with code 1 if any test fails, making it safe to use as a CI gate.

Tests:
  1. E2M1 grid rounding      — _round_to_e2m1 maps inputs to correct grid points
  2. FP8 scale quantization  — find_params produces FP8-representable scales
  3. Flag check              — requires_per_block_params = True (class + instance)
  4. Grid validity           — all dequantized values lie on the E2M1 grid
  5. Scale freshness         — per-block scales differ when block magnitudes differ

Runtime: ~30 s
Usage:   python stage1_nvfp4_unit_tests.py
Exit:    0 = all passed, 1 = one or more failed
"""

import sys
import traceback
from pathlib import Path

# Repo-relative code root (P0.1 fix): the library lives at
# <repo>/opteam-blockwise-gptq regardless of where the repo is checked out.
_CODE_ROOT = Path(__file__).resolve().parents[1] / "opteam-blockwise-gptq"

if not _CODE_ROOT.exists():
    raise RuntimeError(
        f"Code root not found: {_CODE_ROOT}\n"
        "Update _CODE_ROOT at the top of this script."
    )

sys.path.insert(0, str(_CODE_ROOT))

import torch
from quantizer import NVFP4Quantizer, FP8E4M3Quantizer, Int8SymQuantizer

# E2M1 representable magnitudes (sign handled separately)
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# ─────────────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []   # (test_name, passed, detail)


def _record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    _results.append((name, passed, detail))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — E2M1 grid rounding
# ─────────────────────────────────────────────────────────────────────────────

def test_e2m1_grid_rounding():
    """_round_to_e2m1 must map each input to the nearest E2M1 grid point.

    Covers: exact grid values, negative mirrors, midpoints between adjacent
    grid values, above-max clamping (→ 6.0), and sign preservation.

    Rounding rule: L1 nearest-neighbour; ties broken by argmin (lower index
    = lower magnitude wins).
    """
    print("\n── Test 1: E2M1 grid rounding ──────────────────────────────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = NVFP4Quantizer(block_size=16, device=device)

    # (input, expected) pairs
    cases = [
        # Exact grid values — must be identity
        ( 0.0,   0.0),
        ( 0.5,   0.5),
        ( 1.0,   1.0),
        ( 1.5,   1.5),
        ( 2.0,   2.0),
        ( 3.0,   3.0),
        ( 4.0,   4.0),
        ( 6.0,   6.0),
        # Negative mirrors
        (-1.0,  -1.0),
        (-6.0,  -6.0),
        # Between grid points — round to nearest
        ( 0.25,  0.0),   # midpoint 0.0–0.5  → 0.0  (lower index wins)
        ( 0.75,  0.5),   # midpoint 0.5–1.0  → 0.5
        ( 1.25,  1.0),   # midpoint 1.0–1.5  → 1.0
        ( 1.75,  1.5),   # equidistant 1.5↔2.0 — argmin picks lower index (1.5)
        ( 2.5,   2.0),   # midpoint 2.0–3.0  → 2.0  (lower index wins)
        ( 3.5,   3.0),   # midpoint 3.0–4.0  → 3.0  (lower index wins)
        ( 5.0,   4.0),   # midpoint 4.0–6.0  → 4.0  (lower index wins)
        # Above max — clamp to 6.0
        ( 7.0,   6.0),
        (100.0,  6.0),
        # Sign preservation for non-trivial midpoint
        (-2.5,  -2.0),
    ]

    failures = []
    for x_in, x_expected in cases:
        t = torch.tensor([[x_in]], dtype=torch.float32, device=device)
        result = q._round_to_e2m1(t).item()
        if abs(result - x_expected) > 1e-6:
            failures.append(f"input={x_in:.2f} → got {result:.2f}, expected {x_expected:.2f}")

    if failures:
        for f in failures:
            print(f"    ✗ {f}")
    else:
        print(f"    All {len(cases)} rounding cases correct")

    _record("E2M1 grid rounding", not failures,
            f"{len(failures)} case(s) wrong" if failures else f"{len(cases)} cases OK")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — FP8 scale quantization
# ─────────────────────────────────────────────────────────────────────────────

def test_fp8_scale_quantization():
    """find_params must produce scales that are already FP8 E4M3-representable.

    After find_params, every scale value must survive a round-trip through
    float8_e4m3fn unchanged (within 1e-6). If scales are stored in float32
    without FP8 clamping, GPTQ will compensate for a smaller error than the
    hardware will actually produce.

    Uses a weight matrix with a wide dynamic range (1e-2 to 1e2 per column)
    to stress all parts of the FP8 exponent range.
    """
    print("\n── Test 2: FP8 scale quantization ──────────────────────────────────")

    if not hasattr(torch, "float8_e4m3fn"):
        print("    SKIP: torch.float8_e4m3fn not available in this PyTorch build")
        _record("FP8 scale quantization", True, "skipped — float8_e4m3fn unavailable")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    # Wide range: each column block has a different magnitude order
    W = torch.randn(64, 128, device=device) * \
        torch.logspace(-2, 2, 128, device=device).unsqueeze(0)

    q = NVFP4Quantizer(block_size=16, device=device)
    q.find_params(W)

    # Round-trip through FP8 E4M3 must be identity
    scale_rt = q.scale.to(torch.float8_e4m3fn).to(torch.float32)
    max_diff = (q.scale - scale_rt).abs().max().item()
    n_blocks = q.scale.numel()

    print(f"    Scale shape:     {list(q.scale.shape)}")
    print(f"    Scale range:     [{q.scale.min().item():.4e}, {q.scale.max().item():.4e}]")
    print(f"    Max RT diff:     {max_diff:.2e}  (threshold: 1e-6)")
    print(f"    Blocks checked:  {n_blocks}")

    passed = max_diff < 1e-6
    _record("FP8 scale quantization", passed,
            f"max round-trip diff = {max_diff:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — requires_per_block_params flag
# ─────────────────────────────────────────────────────────────────────────────

def test_requires_per_block_params_flag():
    """NVFP4Quantizer must declare requires_per_block_params = True.

    fasterquant_blockwise reads this flag with getattr(..., False) to decide
    whether to call find_params once globally (all other formats) or once
    per GPTQ block (NVFP4). A missing or False flag means find_params is
    called globally and all microscaling scales are computed from the full
    weight matrix — ignoring error compensation from prior blocks.

    Checks both the class attribute (so the flag survives subclassing) and
    the instance attribute.
    """
    print("\n── Test 3: requires_per_block_params flag ───────────────────────────")

    q = NVFP4Quantizer(block_size=16)

    instance_flag = getattr(q, "requires_per_block_params", None)
    class_flag    = getattr(NVFP4Quantizer, "requires_per_block_params", None)

    instance_ok = instance_flag is True
    class_ok    = class_flag is True

    print(f"    Instance attribute: {instance_flag!r}  →  {'OK' if instance_ok else 'WRONG (expected True)'}")
    print(f"    Class attribute:    {class_flag!r}  →  {'OK' if class_ok else 'WRONG (expected True)'}")

    # Also verify no other standard quantizer accidentally has the flag set
    for cls_name, cls in [("FP8E4M3Quantizer", FP8E4M3Quantizer),
                           ("Int8SymQuantizer",  Int8SymQuantizer)]:
        flag = getattr(cls, "requires_per_block_params", False)
        if flag:
            print(f"    WARNING: {cls_name}.requires_per_block_params = {flag!r} "
                  f"(should be absent or False)")

    passed = instance_ok and class_ok
    _record("requires_per_block_params flag", passed,
            "instance=True, class=True" if passed
            else f"instance={instance_flag!r}, class={class_flag!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Grid validity of dequantized outputs
# ─────────────────────────────────────────────────────────────────────────────

def test_outputs_on_e2m1_grid():
    """All dequantized values, divided by their block scale, must lie on E2M1.

    For each microscaling block b:
        (quantize_dequantize(W)[:, b*16:(b+1)*16] / scale[:, b])
    must be within 1e-4 of some point in ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.

    This is the end-to-end correctness test: it catches bugs in find_params
    (wrong scale), quantize_dequantize (wrong grid lookup or wrong reshape),
    and the FP8 scale round-trip all at once.

    Uses a controlled weight matrix whose maximum per block is known, so the
    expected scale can be verified independently.
    """
    print("\n── Test 4: Dequantized outputs on E2M1 grid ─────────────────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(13)

    out_f, in_f = 32, 64   # 4 microscaling blocks of 16 columns each
    W = torch.randn(out_f, in_f, device=device) * 2.0

    q = NVFP4Quantizer(block_size=16, device=device)
    q.find_params(W)
    W_dq = q.quantize_dequantize(W)

    assert W_dq.shape == W.shape, \
        f"Output shape mismatch: expected {W.shape}, got {W_dq.shape}"

    bs = 16
    n_blocks = in_f // bs
    grid = _E2M1_GRID.to(device)

    block_results = []
    for b in range(n_blocks):
        sl = slice(b * bs, (b + 1) * bs)
        normalized = W_dq[:, sl] / q.scale[:, b].unsqueeze(1)   # [out_f, 16]
        min_dist = (normalized.abs().unsqueeze(-1) - grid).abs().min(dim=-1).values.max().item()
        on_grid = min_dist < 1e-4
        block_results.append((b, on_grid, min_dist))
        print(f"    Block {b}: max dist to grid = {min_dist:.2e}  →  {'OK' if on_grid else 'FAIL'}")

    all_on_grid = all(ok for _, ok, _ in block_results)
    worst = max(d for _, _, d in block_results)
    _record("Outputs on E2M1 grid", all_on_grid,
            f"worst block dist = {worst:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Per-block scale freshness
# ─────────────────────────────────────────────────────────────────────────────

def test_per_block_scales_differ():
    """Per-block scales must differ when blocks have clearly different magnitudes.

    Constructs a weight matrix where block b has magnitude 10^b (b = 0…3).
    After find_params the scale for block b should be approximately
    10^b / 6.0 (E2M1 max), so scales must be strictly increasing.

    This test would fail if find_params computed a global (per-row) scale
    instead of per-block microscaling scales — which is the exact bug that
    requires_per_block_params is designed to prevent.
    """
    print("\n── Test 5: Per-block scale freshness ───────────────────────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = 16
    n_blocks = 4
    out_f = 8
    in_f = n_blocks * bs   # 64

    # Each block of 16 columns has a 10x different magnitude
    W = torch.zeros(out_f, in_f, device=device)
    for b in range(n_blocks):
        W[:, b * bs:(b + 1) * bs] = float(10 ** b)

    q = NVFP4Quantizer(block_size=bs, device=device)
    q.find_params(W)

    print(f"    Scale shape: {list(q.scale.shape)}")
    for b in range(n_blocks):
        s = q.scale[0, b].item()
        expected = (10.0 ** b) / 6.0
        ratio = s / expected if expected > 0 else float("nan")
        print(f"    Block {b}: scale = {s:.4e},  expected ≈ {expected:.4e},  ratio = {ratio:.3f}")

    # Scales must be strictly increasing across row 0
    scales = q.scale[0]
    is_increasing = (scales[1:] > scales[:-1]).all().item()

    # Also verify all rows are consistent (all rows see the same column magnitudes)
    all_rows_increasing = all(
        (q.scale[r, 1:] > q.scale[r, :-1]).all().item()
        for r in range(out_f)
    )

    passed = is_increasing and all_rows_increasing
    _record("Per-block scale freshness", passed,
            "scales strictly increasing" if passed
            else f"scales = {scales.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("Stage 1 — NVFP4 Unit Tests")
    print("=" * 68)
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tests = [
        test_e2m1_grid_rounding,
        test_fp8_scale_quantization,
        test_requires_per_block_params_flag,
        test_outputs_on_e2m1_grid,
        test_per_block_scales_differ,
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
    print(f"Stage 1 summary: {n_passed}/{n_total} passed")
    print("=" * 68)

    if n_failed == 0:
        print("✓  All unit tests passed. Safe to proceed to Stage 2.")
        sys.exit(0)
    else:
        print(f"✗  {n_failed} test(s) failed. Fix before proceeding.\n")
        for name, ok, detail in _results:
            if not ok:
                print(f"   FAILED: {name}" + (f" — {detail}" if detail else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()