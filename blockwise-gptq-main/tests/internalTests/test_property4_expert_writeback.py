"""
Property 4 Test: Expert write-back transpose correctness
=========================================================

GPT-OSS stores expert weights in matmul convention [in, out] (x @ W),
while GPTQ expects nn.Linear convention [out, in] (W @ x).

gpt_oss_expert_gptq.py handles this with two transpositions:

    Setup:      shim.weight = experts.gate_up_proj[e].T    [out, in]  ← for GPTQ
    Write-back: experts.gate_up_proj[e] = shim.weight.T    [in,  out] ← back to original

If either transpose is wrong, the expert weights are silently corrupted
(quantized in the wrong orientation). The model would still run but produce
wrong outputs — a bug that is hard to catch without this test.

Four sub-tests:

    4a. Setup transpose
        shim.weight must equal experts.gate_up_proj[e].T (float32) for every e.
        Same for down_proj. No sharing between experts (independent copies).

    4b. Write-back transpose (RTN path)
        After quantize_and_writeback_experts with zero calibration samples,
        the RTN fallback applies. Verify:
          - Shape is preserved: [in, out] not [out, in]
          - Values changed (actually quantized)
          - The quantized values match applying RTN directly to the original

    4c. Data isolation between experts
        Expert e's shim must contain expert e's weights, not expert e±1's.
        Tested by giving each expert a unique constant weight matrix.

    4d. GPTQ path write-back
        With real calibration data (add_batch called), verify the GPTQ path
        also writes back correctly in [in, out] orientation.

Usage:
    python tests/test_property4_expert_writeback.py

Exit: 0 = all passed, 1 = any failed
"""

import sys
import math
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict

# ── Paths ──────────────────────────────────────────────────────────────────────

_CODE_ROOT = Path(
    "/home/runara_dgx_spark_1/Itamar/projects"
    "/Block-wise-GPTQ-GPT-OSS-20B-NVFP4"
    "/opteam-blockwise-gptq"
)
if not _CODE_ROOT.exists():
    raise RuntimeError(f"Code root not found: {_CODE_ROOT}")
sys.path.insert(0, str(_CODE_ROOT))

from gpt_oss_expert_gptq import (
    setup_expert_gptq_instances,
    quantize_and_writeback_experts,
)
from quantizer import NVFP4Quantizer


# ── Mock GptOssExperts ─────────────────────────────────────────────────────────

class MockExperts:
    """
    Minimal stand-in for GptOssExperts.

    Only the attributes accessed by setup_expert_gptq_instances and
    quantize_and_writeback_experts are needed:
        gate_up_proj  [n_exp, hidden, gate_up_out]
        down_proj     [n_exp, intermediate, hidden]

    Weights are plain float32 tensors (nn.Parameter in the real model,
    but .data access works on plain tensors too).
    """
    def __init__(
        self,
        n_experts:    int,
        hidden:       int,
        gate_up_out:  int,
        seed:         int = 0,
    ):
        torch.manual_seed(seed)
        # Use bfloat16 to match the real model dtype
        self.gate_up_proj = torch.randn(n_experts, hidden, gate_up_out,
                                        dtype=torch.bfloat16)
        self.down_proj    = torch.randn(n_experts, hidden, hidden,
                                        dtype=torch.bfloat16)


# ── Sub-tests ──────────────────────────────────────────────────────────────────

def test_4a_setup_transpose(n_experts: int, hidden: int, gate_up_out: int) -> Dict:
    """
    After setup_expert_gptq_instances:
      shim_gu[e].layer.weight  ==  gate_up_proj[e].T.float()   for all e
      shim_dn[e].layer.weight  ==  down_proj[e].T.float()      for all e
    Also checks that each expert's shim holds an independent copy
    (modifying shim[0].weight must not affect shim[1].weight).
    """
    experts = MockExperts(n_experts, hidden, gate_up_out)
    gate_up_gptq, down_gptq = setup_expert_gptq_instances(
        experts, quant_format="nvfp4", device="cpu"
    )

    errors = []

    for e in range(n_experts):
        # ── gate_up ──────────────────────────────────────────────────────
        expected_gu = experts.gate_up_proj[e].float().T   # [gate_up_out, hidden]
        actual_gu   = gate_up_gptq[e].layer.weight.data   # should be [gate_up_out, hidden]

        if actual_gu.shape != expected_gu.shape:
            errors.append(f"  gate_up[{e}] shape: got {actual_gu.shape}, "
                          f"expected {expected_gu.shape}")
            continue

        max_diff = (actual_gu - expected_gu).abs().max().item()
        if max_diff > 1e-4:
            errors.append(f"  gate_up[{e}] max_diff={max_diff:.2e} (need < 1e-4)")

        # ── down ─────────────────────────────────────────────────────────
        expected_dn = experts.down_proj[e].float().T      # [hidden, hidden]
        actual_dn   = down_gptq[e].layer.weight.data

        if actual_dn.shape != expected_dn.shape:
            errors.append(f"  down[{e}] shape: got {actual_dn.shape}, "
                          f"expected {expected_dn.shape}")
            continue

        max_diff = (actual_dn - expected_dn).abs().max().item()
        if max_diff > 1e-4:
            errors.append(f"  down[{e}] max_diff={max_diff:.2e} (need < 1e-4)")

    # ── Data isolation: modifying shim[0] must not affect shim[1] ────────
    if n_experts >= 2:
        w1_before = gate_up_gptq[1].layer.weight.data.clone()
        gate_up_gptq[0].layer.weight.data.fill_(999.0)
        w1_after  = gate_up_gptq[1].layer.weight.data
        if not torch.allclose(w1_before, w1_after):
            errors.append("  Data leakage: modifying shim[0] affected shim[1]")

    passed = len(errors) == 0
    return {
        "name":   "4a_setup_transpose",
        "passed": passed,
        "detail": "All shim weights correctly transposed & isolated" if passed
                  else f"{len(errors)} error(s):\n" + "\n".join(errors),
    }


def test_4b_writeback_rtn(n_experts: int, hidden: int, gate_up_out: int) -> Dict:
    """
    RTN path (nsamples == 0 for all experts):
      1. Shape preserved: gate_up_proj[e] stays [hidden, gate_up_out]
      2. Values changed: the weights are actually quantized
      3. Correct orientation: verify by checking the write-back matches
         applying RTN directly to the original [in, out] weight
    """
    experts = MockExperts(n_experts, hidden, gate_up_out, seed=42)
    W_gu_orig = experts.gate_up_proj.clone()   # [n, hidden, gate_up_out]
    W_dn_orig = experts.down_proj.clone()       # [n, hidden, hidden]

    gate_up_gptq, down_gptq = setup_expert_gptq_instances(
        experts, quant_format="nvfp4", device="cpu"
    )
    # Do NOT call add_batch → nsamples == 0 → RTN fallback
    quantize_and_writeback_experts(
        experts, gate_up_gptq, down_gptq, blocksize=128, percdamp=0.01
    )

    errors = []

    for e in range(n_experts):
        # ── Shape preservation ────────────────────────────────────────────
        if experts.gate_up_proj[e].shape != W_gu_orig[e].shape:
            errors.append(f"  gate_up[{e}] shape changed: "
                          f"{W_gu_orig[e].shape} → {experts.gate_up_proj[e].shape}")

        if experts.down_proj[e].shape != W_dn_orig[e].shape:
            errors.append(f"  down[{e}] shape changed: "
                          f"{W_dn_orig[e].shape} → {experts.down_proj[e].shape}")

        # ── Values changed (quantization happened) ────────────────────────
        if torch.allclose(experts.gate_up_proj[e].float(),
                          W_gu_orig[e].float(), atol=1e-3):
            errors.append(f"  gate_up[{e}]: values unchanged after RTN — "
                          "quantization did not run")

        # ── Correct orientation: RTN on [in, out] should match ────────────
        # Apply RTN directly in [in, out] orientation (as the real code does)
        q = NVFP4Quantizer(block_size=16, device="cpu")
        W_ref = W_gu_orig[e].float().T           # → [out, in]  (as shim sees it)
        q.find_params(W_ref)
        W_ref_q = q.quantize_dequantize(W_ref).T  # → [in, out]  (write-back orientation)

        max_diff = (experts.gate_up_proj[e].float() - W_ref_q.to(torch.float32)).abs().max().item()
        if max_diff > 1e-3:
            errors.append(f"  gate_up[{e}] write-back doesn't match direct RTN "
                          f"(max_diff={max_diff:.2e})")

    passed = len(errors) == 0
    return {
        "name":   "4b_writeback_rtn",
        "passed": passed,
        "detail": f"All {n_experts} experts: shape preserved, values quantized, "
                  "orientation correct" if passed
                  else f"{len(errors)} error(s):\n" + "\n".join(errors),
    }


def test_4c_data_isolation(n_experts: int, hidden: int, gate_up_out: int) -> Dict:
    """
    Each expert's shim must hold its own weight, not a neighbour's.

    Assign expert e a weight matrix of all e+1 (so expert 0 = all 1s,
    expert 1 = all 2s, etc.). After setup, verify shim[e].weight is
    proportional to e+1, not any other value.
    """
    experts = MockExperts(n_experts, hidden, gate_up_out)

    for e in range(n_experts):
        experts.gate_up_proj[e] = torch.full(
            (hidden, gate_up_out), float(e + 1), dtype=torch.bfloat16
        )
        experts.down_proj[e] = torch.full(
            (hidden, hidden), float(e + 1), dtype=torch.bfloat16
        )

    gate_up_gptq, down_gptq = setup_expert_gptq_instances(
        experts, quant_format="nvfp4", device="cpu"
    )

    errors = []
    for e in range(n_experts):
        expected_val = float(e + 1)

        # shim weight should be the transpose, so all values = e+1
        gu_vals = gate_up_gptq[e].layer.weight.data.float().unique()
        if gu_vals.numel() != 1 or abs(gu_vals.item() - expected_val) > 1e-3:
            errors.append(f"  gate_up shim[{e}] has wrong values: "
                          f"unique={gu_vals.tolist()}, expected all={expected_val}")

        dn_vals = down_gptq[e].layer.weight.data.float().unique()
        if dn_vals.numel() != 1 or abs(dn_vals.item() - expected_val) > 1e-3:
            errors.append(f"  down shim[{e}] has wrong values: "
                          f"unique={dn_vals.tolist()}, expected all={expected_val}")

    passed = len(errors) == 0
    return {
        "name":   "4c_data_isolation",
        "passed": passed,
        "detail": f"All {n_experts} experts contain only their own weights" if passed
                  else f"{len(errors)} error(s):\n" + "\n".join(errors),
    }


def test_4d_gptq_writeback(n_experts: int, hidden: int, gate_up_out: int,
                            n_samples: int) -> Dict:
    """
    GPTQ path (add_batch called): verify write-back preserves [in, out] shape
    and that the quantized weight differs from the original (GPTQ ran).
    """
    experts = MockExperts(n_experts, hidden, gate_up_out, seed=7)
    W_gu_orig = experts.gate_up_proj.clone()

    gate_up_gptq, down_gptq = setup_expert_gptq_instances(
        experts, quant_format="nvfp4", device="cpu"
    )

    # Feed calibration data to all experts
    for e in range(n_experts):
        X_gu = torch.randn(n_samples, hidden)          # input to gate_up
        X_dn = torch.randn(n_samples, gate_up_out // 2)  # input to down (intermediate)
        gate_up_gptq[e].add_batch(X_gu, None)
        down_gptq[e].add_batch(X_dn, None)

    quantize_and_writeback_experts(
        experts, gate_up_gptq, down_gptq, blocksize=128, percdamp=0.01
    )

    errors = []
    for e in range(n_experts):
        # Shape must be preserved
        if experts.gate_up_proj[e].shape != W_gu_orig[e].shape:
            errors.append(f"  gate_up[{e}] shape changed after GPTQ write-back: "
                          f"{W_gu_orig[e].shape} → {experts.gate_up_proj[e].shape}")

        # Values must have changed
        if torch.allclose(experts.gate_up_proj[e].float(),
                          W_gu_orig[e].float(), atol=1e-3):
            errors.append(f"  gate_up[{e}]: values unchanged after GPTQ — "
                          "quantization did not run")

    passed = len(errors) == 0
    return {
        "name":   "4d_gptq_writeback",
        "passed": passed,
        "detail": f"All {n_experts} experts: shape preserved, values quantized (GPTQ)" if passed
                  else f"{len(errors)} error(s):\n" + "\n".join(errors),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 64)
    print("Property 4: Expert write-back transpose correctness")
    print("=" * 64)
    print()

    # Use GPT-OSS dimensions but fewer experts (4 instead of 32) for speed
    N_EXP      = 4
    HIDDEN     = 2880
    GATE_UP    = 5760   # 2 × HIDDEN (fused gate + up)
    N_SAMPLES  = 64

    print(f"  Experts    : {N_EXP}  (GPT-OSS has 32; using {N_EXP} for speed)")
    print(f"  hidden     : {HIDDEN}")
    print(f"  gate_up_out: {GATE_UP}  (fused gate+up)")
    print(f"  n_samples  : {N_SAMPLES}  (calibration tokens for 4d)")
    print()

    tests = [
        test_4a_setup_transpose(N_EXP, HIDDEN, GATE_UP),
        test_4b_writeback_rtn(N_EXP, HIDDEN, GATE_UP),
        test_4c_data_isolation(N_EXP, HIDDEN, GATE_UP),
        test_4d_gptq_writeback(N_EXP, HIDDEN, GATE_UP, N_SAMPLES),
    ]

    all_passed = True
    for r in tests:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"[{status}]  {r['name']}")
        print(f"         {r['detail']}")
        print()
        if not r["passed"]:
            all_passed = False

    print("=" * 64)
    if all_passed:
        print("✓  All Property 4 tests passed.")
        print("   Expert transpose convention is correctly implemented.")
    else:
        print("✗  FAILED. Expert write-back has a transpose bug.")
        print("   Weights are being quantized in the wrong orientation.")
    print("=" * 64)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())