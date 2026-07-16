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
    call_counter: Optional[dict] = None,
):
    """Monkey-patch GptOssExperts.forward to collect per-expert Hessians.

    The patched forward reimplements the explicit expert loop of the pinned
    Transformers implementation (verified against transformers 5.14.0,
    ``models/gpt_oss/modeling_gpt_oss.py::GptOssExperts.forward``), inserting
    add_batch() calls at the two projection inputs. For each hit expert e:
      1. gate_up_gptq[e].add_batch(current_state)   — input to gate_up
      2. gate_up + clamped SwiGLU (matches GptOssExperts._apply_gate)
      3. down_gptq[e].add_batch(gated_output)        — input to down
      4. down projection, weighted by routing score, index_add into output

    Routing contract (P0.2 fix)
    ---------------------------
    Two Transformers variants exist for the tensors GptOssExperts.forward
    receives; they are distinguished at call time by shape:

      top-k variant (transformers >= 4.56 / 5.x, incl. pinned 5.14.0):
          hidden_states   [num_tokens, hidden]      (already flat)
          router_indices  [num_tokens, top_k]       expert IDs
          routing_weights [num_tokens, top_k]       softmax over top-k values
          → weight lookup is routing_weights[token_idx, TOP_K_POS]

      dense variant (early transformers 4.55.x):
          routing_weights [num_tokens, num_experts] scatter of top-k softmax
          → weight lookup is routing_weights[token_idx, EXPERT_ID]

    The old code hard-coded a third, incorrect hybrid: it derived
    num_experts from routing_weights.shape[1] (= top_k on the pinned
    version, crashing one_hot for any expert ID >= top_k+1) and indexed
    routing_weights by expert ID.

    The expert mask uses num_classes=num_experts, exactly like the pinned
    implementation (no +1 padding slot in 5.14.0).

    Args:
        experts:       GptOssExperts instance to patch.
        gate_up_gptq:  From setup_expert_gptq_instances (any objects with
                       .add_batch — GPTQ or _GptqH).
        down_gptq:     From setup_expert_gptq_instances.
        call_counter:  Optional dict; ``call_counter["n"]`` is incremented on
                       every patched-forward invocation. Callers use this to
                       fail loudly if the patch was bypassed (e.g. by a fused
                       MegaBlocks kernel forward — GptOssMLP is decorated with
                       @use_kernel_forward_from_hub, which can skip
                       experts.forward entirely if kernelization is enabled).

    Returns:
        original_forward: the original bound method.
        Restore after calibration: ``experts.forward = original_forward``.
    """
    original_forward = experts.forward  # save bound method

    def _patched_forward(hidden_states, router_indices=None, routing_weights=None):
        if call_counter is not None:
            call_counter["n"] = call_counter.get("n", 0) + 1

        num_experts = experts.gate_up_proj.shape[0]

        # hidden_states arrives flat [num_tokens, hidden] on the pinned
        # version (GptOssMLP flattens before calling). Preserve whatever
        # shape we were given.
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, experts.hidden_size)

        # ── Detect the routing-weights contract by shape (see docstring) ──
        if routing_weights.shape[-1] == num_experts:
            dense_weights = True            # dense variant: index by expert ID
        elif routing_weights.shape == router_indices.shape:
            dense_weights = False           # top-k variant: index by position
        else:
            raise RuntimeError(
                f"Unrecognized GPT-OSS routing contract: "
                f"routing_weights {tuple(routing_weights.shape)}, "
                f"router_indices {tuple(router_indices.shape)}, "
                f"num_experts {num_experts}. Inspect the installed "
                f"transformers GptOssExperts.forward and update "
                f"patch_expert_forward."
            )

        next_states = torch.zeros_like(flat)

        with torch.no_grad():
            # [num_tokens, top_k, num_experts] → [num_experts, top_k, num_tokens]
            mask = F.one_hot(router_indices, num_classes=num_experts)
            mask = mask.permute(2, 1, 0)
            expert_hit = (mask.sum(dim=(-1, -2)) > 0).nonzero()

        for idx_tensor in expert_hit:
            e = idx_tensor[0].item()

            with torch.no_grad():
                top_k_pos, token_idx = torch.where(mask[e])

            current_state = flat[token_idx]        # [T_e, hidden]

            # ── Hessian: gate_up input ────────────────────────────────────
            gate_up_gptq[e].add_batch(current_state.detach().float(), None)

            # ── forward: gate_up + clamped SwiGLU (== _apply_gate) ────────
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

            # ── forward: down projection + routing weight ─────────────────
            out = gated_output @ experts.down_proj[e] + experts.down_proj_bias[e]
            if dense_weights:
                w = routing_weights[token_idx, e, None]
            else:
                w = routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, (out * w).to(flat.dtype))

        return next_states.view(orig_shape)

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