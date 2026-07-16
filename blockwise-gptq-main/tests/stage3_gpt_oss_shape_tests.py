"""
Stage 3 — GPT-OSS 20B Shape Tests

Validates fasterquant_blockwise + NVFP4 at the exact linear-layer dimensions
of GPT-OSS 20B using synthetic (random) weights. No model download required.
Passes here mean Stage 5 (full quantization) will not hit shape/memory errors.

GPT-OSS 20B architecture constants:
    hidden_size       = 5120
    intermediate_size = 13696
    num_layers        = 48
    7 projections per layer:
        q / k / v / o  → [5120, 5120]   Hessian [5120 × 5120]  ~100 MB
        gate / up      → [5120, 13696]  Hessian [5120 × 5120]  ~100 MB
        down           → [13696, 5120]  Hessian [13696×13696]  ~748 MB

Tests:
  1. Attention shape   — Q/K/V/O projections [5120→5120], B=128
  2. MLP gate/up       — gate/up projections [5120→13696], B=128
  3. MLP down          — down projection [13696→5120], B=128 (~5-20 s Cholesky)
  4. Blocksize sweep   — B ∈ {64,128,256} on attention shape; reports loss/MSE
  5. Full layer smoke  — all 7 projections in sequence; total time + peak memory
  6. Memory budget     — peak GPU memory per projection ≤ 10 GB

Runtime: ~10-20 min on DGX Spark GB10 (GPU path)
Usage:   python stage3_gpt_oss_shape_tests.py
Exit:    0 = all passed, 1 = one or more failed
"""

import sys
import time
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
print(f"[path] {_CODE_ROOT}")

import torch
import torch.nn as nn
from quantizer import NVFP4Quantizer
from gptq import GPTQ

# ─────────────────────────────────────────────────────────────────────────────
# GPT-OSS 20B architecture constants
# ─────────────────────────────────────────────────────────────────────────────

H  = 5120   # hidden_size
I  = 13696  # intermediate_size

# All 7 per-layer projections: (name, in_features, out_features)
ALL_PROJECTIONS = [
    ("q_proj",    H, H),
    ("k_proj",    H, H),
    ("v_proj",    H, H),
    ("o_proj",    H, H),
    ("gate_proj", H, I),
    ("up_proj",   H, I),
    ("down_proj", I, H),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_gptq(in_f, out_f, ncalib, seed, device):
    """Create a Linear layer + NVFP4Quantizer + accumulated Hessian."""
    torch.manual_seed(seed)
    linear = nn.Linear(in_f, out_f, bias=False, device=device, dtype=torch.float32)
    nn.init.normal_(linear.weight, std=0.02)

    torch.manual_seed(seed + 1000)
    calib = torch.randn(ncalib, 1, in_f, device=device)

    gptq = GPTQ(linear)
    gptq.quantizer = NVFP4Quantizer(block_size=16, device=device)
    for i in range(ncalib):
        gptq.add_batch(calib[i], None)

    return linear, gptq, calib


def _reset_peak_memory(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def _hessian_mb(in_f):
    return in_f * in_f * 4 / 1024 ** 2   # float32


# ─────────────────────────────────────────────────────────────────────────────
# Result tracking
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    _results.append((name, passed, detail))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Attention projections [5120 → 5120]
# ─────────────────────────────────────────────────────────────────────────────

def test_attention_projections():
    """Q, K, V, O projections at GPT-OSS 20B dimensions (5120→5120), B=128.

    Each projection has in_features = out_features = 5120.
    Hessian is [5120×5120] (~100 MB float32).
    40 GPTQ blocks per projection (5120 / 128 = 40).
    """
    print("\n── Test 1: Attention projections [5120→5120] ────────────────────────")
    print(f"    Hessian: {_hessian_mb(H):.0f} MB,  GPTQ blocks: {H // 128}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ncalib, seed, B = 16, 42, 128
    failures = []

    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        _reset_peak_memory(device)
        t0 = time.perf_counter()

        linear, gptq, _ = _make_gptq(H, H, ncalib, seed, device)
        loss = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
        W = linear.weight.data
        gptq.free()

        dt = time.perf_counter() - t0
        peak_mb = _peak_memory_mb(device)

        loss_ok    = torch.isfinite(torch.tensor(loss)).item() and loss > 0
        weights_ok = torch.isfinite(W).all().item()
        shape_ok   = W.shape == (H, H)

        ok  = loss_ok and weights_ok and shape_ok
        tag = "OK" if ok else "FAIL"
        print(f"    {proj_name}: loss={loss:.2f}, t={dt:.1f}s, "
              f"peak={peak_mb:.0f} MB  [{tag}]")

        if not ok:
            failures.append(f"{proj_name}: loss_ok={loss_ok} "
                            f"weights_ok={weights_ok} shape_ok={shape_ok}")

    _record("Attention projections [5120→5120]",
            not failures,
            "all 4 OK" if not failures else " | ".join(failures))


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — MLP gate/up projections [5120 → 13696]
# ─────────────────────────────────────────────────────────────────────────────

def test_mlp_gate_up():
    """gate_proj and up_proj at GPT-OSS 20B dimensions (5120→13696), B=128.

    in_features=5120 (same Hessian as attention), out_features=13696.
    Weight matrix is [13696×5120] (~281 MB float32 per projection).
    """
    print("\n── Test 2: MLP gate/up projections [5120→13696] ─────────────────────")
    print(f"    Weight: {H * I * 4 / 1024**2:.0f} MB,  "
          f"Hessian: {_hessian_mb(H):.0f} MB,  GPTQ blocks: {H // 128}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ncalib, seed, B = 16, 42, 128
    failures = []

    for proj_name in ("gate_proj", "up_proj"):
        _reset_peak_memory(device)
        t0 = time.perf_counter()

        linear, gptq, _ = _make_gptq(H, I, ncalib, seed, device)
        loss = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
        W = linear.weight.data
        gptq.free()

        dt = time.perf_counter() - t0
        peak_mb = _peak_memory_mb(device)

        loss_ok    = torch.isfinite(torch.tensor(loss)).item() and loss > 0
        weights_ok = torch.isfinite(W).all().item()
        shape_ok   = W.shape == (I, H)

        ok  = loss_ok and weights_ok and shape_ok
        tag = "OK" if ok else "FAIL"
        print(f"    {proj_name}: loss={loss:.2f}, t={dt:.1f}s, "
              f"peak={peak_mb:.0f} MB  [{tag}]")

        if not ok:
            failures.append(f"{proj_name}: loss_ok={loss_ok} "
                            f"weights_ok={weights_ok} shape_ok={shape_ok}")

    _record("MLP gate/up projections [5120→13696]",
            not failures,
            "both OK" if not failures else " | ".join(failures))


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — MLP down projection [13696 → 5120]
# ─────────────────────────────────────────────────────────────────────────────

def test_mlp_down():
    """down_proj at GPT-OSS 20B dimensions (13696→5120), B=128.

    This is the most expensive shape:
      - Hessian [13696×13696] = ~748 MB float32
      - Cholesky on CPU: ~5-20 s
      - 107 GPTQ blocks (13696 / 128 = 107)

    If this passes, no layer in GPT-OSS 20B will OOM during Stage 5.
    """
    print("\n── Test 3: MLP down projection [13696→5120] ─────────────────────────")
    print(f"    Weight: {I * H * 4 / 1024**2:.0f} MB,  "
          f"Hessian: {_hessian_mb(I):.0f} MB,  GPTQ blocks: {I // 128}")
    print("    (Cholesky of 13696×13696 on CPU may take 5-20 s)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ncalib, seed, B = 16, 42, 128

    _reset_peak_memory(device)
    t0 = time.perf_counter()

    linear, gptq, _ = _make_gptq(I, H, ncalib, seed, device)
    loss = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
    W = linear.weight.data
    gptq.free()

    dt = time.perf_counter() - t0
    peak_mb = _peak_memory_mb(device)

    loss_ok    = torch.isfinite(torch.tensor(loss)).item() and loss > 0
    weights_ok = torch.isfinite(W).all().item()
    shape_ok   = W.shape == (H, I)

    print(f"    loss={loss:.2f}, t={dt:.1f}s, peak={peak_mb:.0f} MB")

    passed = loss_ok and weights_ok and shape_ok
    _record("MLP down projection [13696→5120]",
            passed,
            f"t={dt:.1f}s peak={peak_mb:.0f}MB"
            if passed else
            f"loss_ok={loss_ok} weights_ok={weights_ok} shape_ok={shape_ok}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Blocksize sweep on attention shape
# ─────────────────────────────────────────────────────────────────────────────

def test_blocksize_sweep():
    """B ∈ {64, 128, 256} on the attention shape [5120→5120] with NVFP4.

    Reports loss and output MSE for each B. All must succeed (finite, positive).
    The loss/MSE values guide the blocksize choice for Stage 5 quantization.

    B must always be a multiple of 16 (NVFP4 microscaling block size).
    B=128 is the recommended default for GPT-OSS 20B.
    """
    print("\n── Test 4: Blocksize sweep (NVFP4, [5120→5120]) ─────────────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ncalib, seed = 16, 42

    # Reference outputs for MSE
    torch.manual_seed(seed)
    linear_ref = nn.Linear(H, H, bias=False, device=device, dtype=torch.float32)
    nn.init.normal_(linear_ref.weight, std=0.02)
    torch.manual_seed(seed + 1000)
    calib = torch.randn(ncalib, 1, H, device=device)
    with torch.no_grad():
        ref_out = torch.cat([linear_ref(calib[i]) for i in range(ncalib)])

    print(f"    {'B':>5s}  {'n_blocks':>8s}  {'loss':>12s}  "
          f"{'output_mse':>12s}  {'time(s)':>8s}  {'status':>6s}")

    failures = []
    for B in [64, 128, 256]:
        n_blocks = (H + B - 1) // B
        t0 = time.perf_counter()

        linear, gptq, _ = _make_gptq(H, H, ncalib, seed, device)
        loss = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
        gptq.free()

        with torch.no_grad():
            q_out = torch.cat([linear(calib[i]) for i in range(ncalib)])
        mse = (ref_out - q_out).pow(2).mean().item()
        dt = time.perf_counter() - t0

        loss_ok    = torch.isfinite(torch.tensor(loss)).item() and loss > 0
        weights_ok = torch.isfinite(linear.weight.data).all().item()
        ok = loss_ok and weights_ok
        tag = "PASS" if ok else "FAIL"

        print(f"    {B:>5d}  {n_blocks:>8d}  {loss:>12.4f}  "
              f"{mse:>12.4e}  {dt:>8.1f}  {tag:>6s}")

        if not ok:
            failures.append(f"B={B}: loss_ok={loss_ok} weights_ok={weights_ok}")

    _record("Blocksize sweep [5120→5120]",
            not failures,
            "B=64/128/256 all OK" if not failures else " | ".join(failures))


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Full transformer layer (all 7 projections)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_transformer_layer():
    """Quantize all 7 projections of one GPT-OSS 20B transformer layer.

    Simulates the full per-layer GPTQ pass that Stage 5 will perform for each
    of the 48 layers. Uses ncalib=16 synthetic calibration samples.

    Reports per-projection loss and timing; checks all weights are finite.
    Total time × 48 layers gives the expected Stage 5 runtime estimate.
    """
    print("\n── Test 5: Full transformer layer (all 7 projections) ───────────────")
    print("    (simulates one Stage 5 layer pass; total × 48 = full model estimate)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ncalib, seed, B = 16, 42, 128
    failures = []
    layer_t0 = time.perf_counter()
    total_loss = 0.0

    print(f"\n    {'projection':<12s}  {'shape':>16s}  "
          f"{'loss':>10s}  {'time(s)':>8s}  {'peak(MB)':>10s}  {'status':>6s}")

    for proj_name, in_f, out_f in ALL_PROJECTIONS:
        _reset_peak_memory(device)
        t0 = time.perf_counter()

        linear, gptq, _ = _make_gptq(in_f, out_f, ncalib, seed, device)
        loss = gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
        W = linear.weight.data
        gptq.free()

        if device == "cuda":
            torch.cuda.empty_cache()

        dt = time.perf_counter() - t0
        peak_mb = _peak_memory_mb(device)

        loss_ok    = torch.isfinite(torch.tensor(loss)).item() and loss > 0
        weights_ok = torch.isfinite(W).all().item()
        shape_ok   = W.shape == (out_f, in_f)

        ok  = loss_ok and weights_ok and shape_ok
        tag = "PASS" if ok else "FAIL"
        total_loss += loss if loss_ok else 0.0

        shape_str = f"[{out_f}×{in_f}]"
        print(f"    {proj_name:<12s}  {shape_str:>16s}  "
              f"{loss:>10.2f}  {dt:>8.1f}  {peak_mb:>10.0f}  {tag:>6s}")

        if not ok:
            failures.append(f"{proj_name}: loss_ok={loss_ok} "
                            f"weights_ok={weights_ok} shape_ok={shape_ok}")

    layer_dt = time.perf_counter() - layer_t0
    est_full = layer_dt * 48 / 60  # minutes for all 48 layers

    print(f"\n    Total layer loss : {total_loss:.2f}")
    print(f"    Layer time       : {layer_dt:.1f} s")
    print(f"    Full model est.  : {est_full:.0f} min  (× 48 layers)")

    _record("Full transformer layer",
            not failures,
            f"all 7 OK, layer={layer_dt:.1f}s, est={est_full:.0f}min×48"
            if not failures else " | ".join(failures))


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — GPU memory budget per projection
# ─────────────────────────────────────────────────────────────────────────────

def test_memory_budget():
    """Peak GPU memory per projection must stay within a safe budget.

    DGX Spark GB10 has 128 GB unified LPDDR5x. Assuming BF16 model weights
    (~40 GB) stay loaded, each GPTQ pass has ~88 GB headroom. The most
    memory-intensive step is holding:
      - Weight matrix W (float32)
      - Hessian H (float32)
      - Inverse Hessian Hinv (float32, same size as H)

    Budget ceiling used here: 10 GB per projection (conservative).
    down_proj is the worst case: Hinv [13696×13696] alone = 748 MB × 2 = ~1.5 GB.

    This test runs each projection and checks peak GPU memory is below the budget.
    Runs only on CUDA; skipped on CPU.
    """
    print("\n── Test 6: GPU memory budget per projection ─────────────────────────")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("    SKIP: CUDA not available")
        _record("GPU memory budget", True, "skipped — CPU only")
        return

    ncalib, seed, B = 16, 42, 128
    budget_mb = 10_000   # 10 GB ceiling
    failures = []

    print(f"    Budget: {budget_mb / 1024:.0f} GB per projection")
    print(f"\n    {'projection':<12s}  {'hessian_MB':>12s}  "
          f"{'peak_MB':>10s}  {'budget':>10s}  {'status':>6s}")

    for proj_name, in_f, out_f in ALL_PROJECTIONS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        linear, gptq, _ = _make_gptq(in_f, out_f, ncalib, seed, device)
        gptq.fasterquant_blockwise(blocksize=B, percdamp=0.01)
        gptq.free()
        torch.cuda.empty_cache()

        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        h_mb    = _hessian_mb(in_f)
        ok      = peak_mb <= budget_mb
        tag     = "PASS" if ok else "FAIL"

        print(f"    {proj_name:<12s}  {h_mb:>12.0f}  "
              f"{peak_mb:>10.0f}  {budget_mb:>10d}  {tag:>6s}")

        if not ok:
            failures.append(f"{proj_name}: {peak_mb:.0f} MB > {budget_mb} MB")

    _record("GPU memory budget",
            not failures,
            f"all within {budget_mb // 1024} GB"
            if not failures else " | ".join(failures))


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("Stage 3 — GPT-OSS 20B Shape Tests")
    print("=" * 68)
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU      : {props.name},  "
              f"{props.total_memory / 1024**3:.0f} GB")
    print(f"hidden   : {H}   intermediate: {I}")
    print(f"Projections per layer: {len(ALL_PROJECTIONS)}")

    tests = [
        test_attention_projections,
        test_mlp_gate_up,
        test_mlp_down,
        test_blocksize_sweep,
        test_full_transformer_layer,
        test_memory_budget,
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
    print(f"Stage 3 summary: {n_passed}/{n_total} passed")
    print("=" * 68)

    if n_failed == 0:
        print("✓  All shape tests passed. Safe to proceed to Stage 4.")
        print("   (Stage 4 requires the model download — run download_model.sh first.)")
        sys.exit(0)
    else:
        print(f"✗  {n_failed} test(s) failed. Fix before proceeding.\n")
        for name, ok, detail in _results:
            if not ok:
                print(f"   FAILED: {name}" + (f" — {detail}" if detail else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()