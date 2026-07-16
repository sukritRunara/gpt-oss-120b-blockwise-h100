"""
GPT-OSS Expert GPTQ Quantization
=================================

GPT-OSS stores all expert weights as batched tensors on a single
GptOssExperts module rather than as nn.ModuleList of nn.Linear modules.
This prevents find_layers() from discovering them, so they are skipped
by the standard GPTQ layer loop.

This module provides three functions that implement full GPTQ quantization
for those expert weights within the existing layer-sequential framework
of gptq_quantize_model():

    setup_expert_gptq_instances
        Creates GPTQ instances with temporary nn.Linear shims for each
        of the N_e × 2 expert projections (gate_up + down per expert).
        The batched tensors are not modified until Phase 3.

    patch_expert_forward
        Monkey-patches GptOssExperts.forward so that during the calibration
        pass it intercepts per-expert activations and calls add_batch() on
        the appropriate GPTQ instance. Returns the original forward so
        the caller can restore it after calibration.

    quantize_and_writeback_experts
        Runs fasterquant_blockwise on every expert shim and copies the
        quantized weights back into the batched tensors in-place.

Weight storage convention
--------------------------
GptOssExperts uses the matmul convention (x @ W), NOT the nn.Linear
convention (W @ x):

    gate_up_proj : [num_experts, hidden,        intermediate*2]
    down_proj    : [num_experts, intermediate,  hidden]

i.e. dim-1 is in_features, dim-2 is out_features.
GPTQ expects weight[out_features, in_features] (nn.Linear convention).
The shims hold W.T; quantize_and_writeback transposes back on copy.

Integration into gptq_quantize_model() (apply.py)
---------------------------------------------------
Four additions to the per-layer loop — see INTEGRATION POINTS below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from gptq import GPTQ


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_gpt_oss_experts(layer) -> bool:
    """Return True if the transformer layer contains a GptOssExperts module."""
    return (
        hasattr(layer, "mlp")
        and hasattr(layer.mlp, "experts")
        and type(layer.mlp.experts).__name__ == "GptOssExperts"
    )


def _make_quantizer(quant_format: str, device, nvfp4_block_size: int = 16):
    """Construct a fresh quantizer for the given format."""
    from quantizer import NVFP4Quantizer, FP8E4M3Quantizer, Int8SymQuantizer
    if quant_format == "nvfp4":
        return NVFP4Quantizer(block_size=nvfp4_block_size, device=device)
    elif quant_format == "fp8":
        return FP8E4M3Quantizer(device=device)
    elif quant_format == "int8":
        return Int8SymQuantizer(device=device)
    else:
        raise ValueError(
            f"Unsupported quant_format for expert GPTQ: {quant_format!r}. "
            "Supported: 'nvfp4', 'fp8', 'int8'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 1 — Phase 1: setup
# Call once per layer, BEFORE the calibration loop.
# ─────────────────────────────────────────────────────────────────────────────

def setup_expert_gptq_instances(
    experts,
    quant_format: str,
    device,
    nvfp4_block_size: int = 16,
) -> Tuple[Dict[int, GPTQ], Dict[int, GPTQ]]:
    """Create GPTQ instances with nn.Linear shims for all expert projections.

    For expert e:
      gate_up_gptq[e]  →  shim.weight = gate_up_proj[e].T  [gate_up_out, hidden]
      down_gptq[e]     →  shim.weight = down_proj[e].T      [hidden, intermediate]

    The shims hold float32 copies of the transposed weights. The original
    batched tensors (experts.gate_up_proj, experts.down_proj) are untouched
    until quantize_and_writeback_experts() is called.

    Args:
        experts:          GptOssExperts module.
        quant_format:     "nvfp4", "fp8", or "int8".
        device:           Compute device.
        nvfp4_block_size: Microscaling block size (16 for NVFP4 hardware).

    Returns:
        (gate_up_gptq, down_gptq) — dicts mapping expert index → GPTQ instance.
    """
    num_experts  = experts.gate_up_proj.shape[0]
    hidden       = experts.gate_up_proj.shape[1]   # in_features of gate_up
    gate_up_out  = experts.gate_up_proj.shape[2]   # out_features of gate_up
    intermediate = experts.down_proj.shape[1]       # in_features of down

    gate_up_gptq: Dict[int, GPTQ] = {}
    down_gptq:    Dict[int, GPTQ] = {}

    for e in range(num_experts):
        # gate_up_proj[e] is [hidden, gate_up_out]  →  shim weight [gate_up_out, hidden]
        shim_gu = nn.Linear(hidden, gate_up_out, bias=False, dtype=torch.float32)
        shim_gu.weight.data.copy_(experts.gate_up_proj[e].detach().T.float())
        shim_gu = shim_gu.to(device)
        g_gu = GPTQ(shim_gu)
        g_gu.quantizer = _make_quantizer(quant_format, device, nvfp4_block_size)
        gate_up_gptq[e] = g_gu

        # down_proj[e] is [intermediate, hidden]  →  shim weight [hidden, intermediate]
        shim_dn = nn.Linear(intermediate, hidden, bias=False, dtype=torch.float32)
        shim_dn.weight.data.copy_(experts.down_proj[e].detach().T.float())
        shim_dn = shim_dn.to(device)
        g_dn = GPTQ(shim_dn)
        g_dn.quantizer = _make_quantizer(quant_format, device, nvfp4_block_size)
        down_gptq[e] = g_dn

    return gate_up_gptq, down_gptq


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 2 — Phase 2: calibration hook
# Call BEFORE the calibration loop. Restore AFTER.
# ─────────────────────────────────────────────────────────────────────────────

def patch_expert_forward(
    experts,
    gate_up_gptq: Dict[int, GPTQ],
    down_gptq: Dict[int, GPTQ],
):
    """Monkey-patch GptOssExperts.forward to collect per-expert Hessians.

    The patched forward always uses the explicit expert loop (the
    ``if self.training`` path), regardless of device. For each active
    expert e it:
      1. calls gate_up_gptq[e].add_batch(current_state)   — input to gate_up
      2. performs the gate_up + SwiGLU computation
      3. calls down_gptq[e].add_batch(gated_output)        — input to down
      4. performs the down projection and combines outputs

    The patched forward produces numerically identical outputs to the
    original training-mode path (same computation, just with add_batch
    calls inserted).

    Args:
        experts:       GptOssExperts instance to patch.
        gate_up_gptq:  From setup_expert_gptq_instances.
        down_gptq:     From setup_expert_gptq_instances.

    Returns:
        original_forward: the original bound method.
        Restore after calibration: ``experts.forward = original_forward``.
    """
    original_forward = experts.forward  # save bound method

    def _patched_forward(hidden_states, router_indices=None, routing_weights=None):
        batch_size = hidden_states.shape[0]
        flat       = hidden_states.reshape(-1, experts.hidden_size)
        num_exp    = routing_weights.shape[1]

        next_states = torch.zeros_like(flat)

        with torch.no_grad():
            mask = F.one_hot(router_indices, num_classes=num_exp + 1)
            mask = mask.permute(2, 1, 0)           # [num_exp+1, top_k, num_tokens]
            expert_hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()

        for idx_tensor in expert_hit:
            e = idx_tensor[0].item()
            if e == num_exp:   # masking slot — skip
                continue

            with torch.no_grad():
                _, token_idx = torch.where(mask[e])

            current_state = flat[token_idx]        # [T_e, hidden]

            # ── Hessian: gate_up input ────────────────────────────────────
            gate_up_gptq[e].add_batch(current_state.detach().float(), None)

            # ── forward: gate_up + SwiGLU ─────────────────────────────────
            gate_up = (
                current_state @ experts.gate_up_proj[e]
                + experts.gate_up_proj_bias[e]
            )
            gate, up   = gate_up[..., ::2], gate_up[..., 1::2]
            gate        = gate.clamp(max=experts.limit)
            up          = up.clamp(min=-experts.limit, max=experts.limit)
            glu         = gate * torch.sigmoid(gate * experts.alpha)
            gated_output = (up + 1) * glu          # [T_e, intermediate]

            # ── Hessian: down input ───────────────────────────────────────
            down_gptq[e].add_batch(gated_output.detach().float(), None)

            # ── forward: down projection ──────────────────────────────────
            out      = gated_output @ experts.down_proj[e] + experts.down_proj_bias[e]
            weighted = out * routing_weights[token_idx, e, None]
            next_states.index_add_(0, token_idx, weighted.to(flat.dtype))

        return next_states.view(batch_size, -1, experts.hidden_size)

    # Instance-level attribute shadows the class method without affecting
    # other GptOssExperts instances (other layers still have the original).
    experts.forward = _patched_forward
    return original_forward


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 3 — Phase 3: quantize + write-back
# Call AFTER standard linear quantization, BEFORE layer.cpu().
# ─────────────────────────────────────────────────────────────────────────────

def quantize_and_writeback_experts(
    experts,
    gate_up_gptq: Dict[int, GPTQ],
    down_gptq:    Dict[int, GPTQ],
    blocksize:    int   = 128,
    percdamp:     float = 0.01,
) -> Tuple[Dict[int, object], Dict[int, object]]:
    """Run fasterquant_blockwise on every expert shim and copy back in-place.

    For each expert e:
      1. fasterquant_blockwise on gate_up shim (shim.weight is [out, in]).
         Copies result back: experts.gate_up_proj[e] = shim.weight.T  ([in, out]).
      2. Same for down shim.

    Experts with zero calibration signal (nsamples == 0) fall back to
    round-to-nearest (RTN) quantization. In practice, with 512 calibration
    samples and top-4 routing over 32 experts, all experts should receive
    ≥ 64 effective samples.

    Args:
        experts:   GptOssExperts module — weights updated IN-PLACE.
        blocksize: GPTQ block width. For nvfp4 must be a multiple of 16.
        percdamp:  Hessian damping factor.

    Returns:
        (gate_up_losses, down_losses) — dicts mapping expert index to float
        loss, or the string "RTN" for never-activated experts.
    """
    orig_dtype      = experts.gate_up_proj.dtype
    gate_up_losses: Dict[int, object] = {}
    down_losses:    Dict[int, object] = {}
    rtn_count = 0

    num_experts = experts.gate_up_proj.shape[0]

    for e in range(num_experts):

        # ── gate_up ───────────────────────────────────────────────────────
        g_gu = gate_up_gptq[e]
        if g_gu.nsamples == 0:
            _rtn_inplace(experts.gate_up_proj.data[e], g_gu.quantizer)
            gate_up_losses[e] = "RTN"
            rtn_count += 1
        else:
            loss = g_gu.fasterquant_blockwise(blocksize=blocksize, percdamp=percdamp)
            # shim.weight is [gate_up_out, hidden]; transpose → [hidden, gate_up_out]
            experts.gate_up_proj.data[e].copy_(
                g_gu.layer.weight.data.T.to(orig_dtype)
            )
            gate_up_losses[e] = loss
        g_gu.free()

        # ── down ──────────────────────────────────────────────────────────
        g_dn = down_gptq[e]
        if g_dn.nsamples == 0:
            _rtn_inplace(experts.down_proj.data[e], g_dn.quantizer)
            down_losses[e] = "RTN"
        else:
            loss = g_dn.fasterquant_blockwise(blocksize=blocksize, percdamp=percdamp)
            # shim.weight is [hidden, intermediate]; transpose → [intermediate, hidden]
            experts.down_proj.data[e].copy_(
                g_dn.layer.weight.data.T.to(orig_dtype)
            )
            down_losses[e] = loss
        g_dn.free()

    return gate_up_losses, down_losses


def _rtn_inplace(weight_in_out, quantizer):
    """Round-to-nearest fallback for an expert stored in [in, out] convention.

    find_params and quantize_dequantize expect [out_features, in_features],
    so we transpose in, quantize, and transpose back.
    """
    w   = weight_in_out.T.float()          # [out, in]
    quantizer.find_params(w)
    w_q = quantizer.quantize_dequantize(w)
    weight_in_out.copy_(w_q.T.to(weight_in_out.dtype))