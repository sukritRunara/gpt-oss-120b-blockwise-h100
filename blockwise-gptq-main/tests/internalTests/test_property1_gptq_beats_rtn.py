"""
Property 1 Test: GPTQ loss ≤ RTN loss (fundamental guarantee)
==============================================================

Tests the core GPTQ correctness guarantee: after error compensation,
the Hessian-weighted output reconstruction error must be ≤ naive
Round-To-Nearest (RTN) quantization on the same weights.

Metric used: output MSE on calibration data
    MSE = ||X W_orig.T - X W_quantized.T||² / n

This is equivalent to the H-weighted loss:
    L = trace( (W - Q(W))  H  (W - Q(W)).T )
    where H = X.T X / n

If GPTQ cannot beat RTN on a simple toy linear layer, the error
compensation in fasterquant_blockwise is broken — regardless of model
or dataset.

Test configurations include shapes matching GPT-OSS 20B projections:
    q_proj    [2880 → 4096]
    gate_up   [2880 → 5760]
    down      [2880 → 2880]

Usage:
    python tests/test_property1_gptq_beats_rtn.py

Exit: 0 = all passed, 1 = any failed
"""

import sys
import math
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List

# ── Paths ──────────────────────────────────────────────────────────────────────

# Repo-relative code root (P0.1 fix): the library lives at
# <repo>/opteam-blockwise-gptq regardless of where the repo is checked out.
_CODE_ROOT = Path(__file__).resolve().parents[2] / "opteam-blockwise-gptq"
if not _CODE_ROOT.exists():
    raise RuntimeError(f"Code root not found: {_CODE_ROOT}")
sys.path.insert(0, str(_CODE_ROOT))

from gptq import GPTQ
from quantizer import NVFP4Quantizer


# ── Core test function ─────────────────────────────────────────────────────────

def run_single_test(
    in_features:  int,
    out_features: int,
    n_samples:    int,
    blocksize:    int,
    seed:         int,
    percdamp:     float = 0.01,
    device:       str   = "cpu",
) -> Dict:
    """
    Run one GPTQ vs RTN comparison on a toy nn.Linear.

    Returns a dict with:
        mse_gptq    — output MSE after GPTQ quantization
        mse_rtn     — output MSE after RTN quantization
        improvement — (mse_rtn - mse_gptq) / mse_rtn * 100  [%]
        passed      — True if mse_gptq <= mse_rtn
        gptq_loss   — internal loss reported by fasterquant_blockwise
    """
    torch.manual_seed(seed)

    # ── Toy linear layer ──────────────────────────────────────────────────────
    linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    W_orig = linear.weight.data.clone()   # [out_features, in_features]

    # ── Calibration data ──────────────────────────────────────────────────────
    # Gaussian activations — representative of normalised transformer hidden states
    X = torch.randn(n_samples, in_features)   # [n, in]

    # ── GPTQ ──────────────────────────────────────────────────────────────────
    g = GPTQ(linear)
    g.quantizer = NVFP4Quantizer(block_size=16, device=device)
    g.add_batch(X, None)           # builds H = 2/n * X.T @ X (+ damping)

    gptq_loss = g.fasterquant_blockwise(blocksize=blocksize, percdamp=percdamp)
    W_gptq = linear.weight.data.clone()   # quantized + error-compensated
    g.free()

    # ── RTN (no error compensation) ───────────────────────────────────────────
    linear.weight.data.copy_(W_orig)      # restore original weights
    q_rtn = NVFP4Quantizer(block_size=16, device=device)
    q_rtn.find_params(W_orig)
    W_rtn = q_rtn.quantize_dequantize(W_orig)   # [out, in], same dtype

    # ── Output MSE on calibration data ───────────────────────────────────────
    # This is the H-weighted loss without needing to access H directly.
    # MSE_GPTQ = ||X(W_orig - W_gptq).T||² / (n * out)
    # MSE_RTN  = ||X(W_orig - W_rtn ).T||² / (n * out)
    err_gptq = X @ (W_orig - W_gptq).T    # [n, out]
    err_rtn  = X @ (W_orig - W_rtn ).T    # [n, out]

    mse_gptq = err_gptq.pow(2).mean().item()
    mse_rtn  = err_rtn .pow(2).mean().item()

    improvement = (mse_rtn - mse_gptq) / mse_rtn * 100 if mse_rtn > 0 else 0.0
    passed = mse_gptq <= mse_rtn * (1 + 1e-6)   # small tolerance for float noise

    return {
        "mse_gptq":    mse_gptq,
        "mse_rtn":     mse_rtn,
        "improvement": improvement,
        "passed":      passed,
        "gptq_loss":   gptq_loss if not math.isnan(gptq_loss) else float("nan"),
    }


# ── Test configurations ────────────────────────────────────────────────────────

CONFIGS = [
    # label             in    out    n     B
    ("tiny",            64,   128,   64,   16),
    ("small",           256,  512,   128,  64),
    ("q_proj-scale",    2880, 4096,  128,  128),   # GPT-OSS q_proj
    ("gate_up-scale",   2880, 5760,  128,  128),   # GPT-OSS gate_up (per expert)
    ("down-scale",      2880, 2880,  128,  128),   # GPT-OSS down_proj (per expert)
    ("o_proj-scale",    4096, 2880,  128,  128),   # GPT-OSS o_proj
]

SEEDS = [0, 1, 2, 3, 4]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 64)
    print("Property 1: GPTQ loss ≤ RTN loss  (fundamental guarantee)")
    print("=" * 64)
    print(f"  Quantizer : NVFP4 (block_size=16)")
    print(f"  percdamp  : 0.01")
    print(f"  Seeds     : {SEEDS}")
    print()

    all_passed = True

    for label, in_f, out_f, n, B in CONFIGS:
        results: List[Dict] = []
        for seed in SEEDS:
            r = run_single_test(in_f, out_f, n, B, seed)
            results.append(r)

        seeds_passed = [r["passed"] for r in results]
        config_passed = all(seeds_passed)

        avg_mse_gptq   = sum(r["mse_gptq"]    for r in results) / len(results)
        avg_mse_rtn    = sum(r["mse_rtn"]      for r in results) / len(results)
        avg_improvement = sum(r["improvement"] for r in results) / len(results)

        status = "✓ PASS" if config_passed else "✗ FAIL"
        print(f"[{status}]  {label:<20s}  ({in_f}→{out_f}, B={B})")
        print(f"         MSE GPTQ = {avg_mse_gptq:.6f}")
        print(f"         MSE RTN  = {avg_mse_rtn:.6f}")
        print(f"         Improvement = {avg_improvement:.1f}%  (avg over {len(SEEDS)} seeds)")

        if not config_passed:
            failing_seeds = [SEEDS[i] for i, p in enumerate(seeds_passed) if not p]
            for i, seed in enumerate(failing_seeds):
                r = results[SEEDS.index(seed)]
                print(f"         FAILED seed={seed}: "
                      f"mse_gptq={r['mse_gptq']:.6f} > mse_rtn={r['mse_rtn']:.6f}")
            all_passed = False

        print()

    print("=" * 64)
    if all_passed:
        print("✓  All Property 1 tests passed.")
        print("   fasterquant_blockwise error compensation is correct.")
    else:
        print("✗  FAILED. Error compensation in fasterquant_blockwise is broken.")
        print("   GPTQ should always match or beat RTN on the same calibration data.")
    print("=" * 64)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())