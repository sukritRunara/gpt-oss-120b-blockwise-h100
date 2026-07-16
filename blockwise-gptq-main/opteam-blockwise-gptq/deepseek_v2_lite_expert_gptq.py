"""
DeepSeek V2 Lite Expert GPTQ Quantization
==========================================

DeepSeek V2 Lite stores expert weights as individual nn.Linear modules inside
DeepseekV2MLP instances, organized in a ModuleList. Unlike GPT-OSS which uses
batched weight tensors, each expert is a proper nn.Linear — so no transposing,
shim construction, or batched-tensor write-back is needed.

However, because of MoE routing, Hessians for expert linears cannot be
collected by the standard GPTQ calibration loop (which processes full hidden
states). Each expert only sees the subset of tokens routed to it. This module
handles that via forward hooks registered directly on the individual Linear
layers inside each expert — the MoE routing happens above them, so each hook
naturally receives exactly the right token subset without any MoE patching.

Structure (per MoE layer, i.e. layers 1-26):
    layer.mlp                         : DeepseekV2MoE
    layer.mlp.experts                 : ModuleList[64 × DeepseekV2MLP]
    layer.mlp.experts[e].gate_proj    : nn.Linear  (hidden → intermediate)
    layer.mlp.experts[e].up_proj      : nn.Linear  (hidden → intermediate)
    layer.mlp.experts[e].down_proj    : nn.Linear  (intermediate → hidden)
    layer.mlp.shared_experts          : DeepseekV2MLP  (all tokens, no routing)
    layer.mlp.shared_experts.gate_proj / up_proj / down_proj : nn.Linear

Layer 0 has a dense MLP (no MoE) — handled by the standard GPTQ path in
apply.py, not by this module.

Integration into gptq_quantize_model() (apply.py)
--------------------------------------------------
Per-layer loop (layers 1-26 only):

    PHASE 1 — before calibration:
        routed_gptq, shared_gptq = setup_expert_gptq_instances(layer, ...)

    PHASE 2 — register hooks before running calibration batches:
        handles = register_expert_hooks(layer, routed_gptq, shared_gptq)
        # ... run calibration forward passes ...
        for h in handles: h.remove()

    PHASE 3 — after standard linear quantization, before layer.cpu():
        losses = quantize_and_writeback_experts(
            layer, routed_gptq, shared_gptq,
            blocksize=blocksize, percdamp=percdamp,
        )
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from gptq import GPTQ


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_deepseek_v2_moe(layer) -> bool:
    """Return True if this transformer layer contains a DeepseekV2MoE MLP.

    Layer 0 of DeepSeek V2 Lite is a dense MLP (DeepseekV2MLP directly),
    not a MoE wrapper — this returns False for it.
    """
    return (
        hasattr(layer, "mlp")
        and type(layer.mlp).__name__ == "DeepseekV2MoE"
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
            f"Unsupported quant_format: {quant_format!r}. "
            "Supported: 'nvfp4', 'fp8', 'int8'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 1 — Phase 1: setup
# Call once per MoE layer, BEFORE the calibration loop.
# ─────────────────────────────────────────────────────────────────────────────

def setup_expert_gptq_instances(
    layer,
    quant_format: str,
    device,
    nvfp4_block_size: int = 16,
) -> Tuple[
    Dict[int, Tuple[GPTQ, GPTQ, GPTQ]],   # routed: e → (gate, up, down)
    Tuple[GPTQ, GPTQ, GPTQ],               # shared: (gate, up, down)
]:
    """Create GPTQ instances wrapping each expert's nn.Linear modules directly.

    Because DeepSeek V2 Lite experts are standard nn.Linear, GPTQ wraps
    the actual module — no shims or transposing required. When
    fasterquant_blockwise runs, it updates layer.weight in-place automatically.

    Args:
        layer:            Transformer layer containing DeepseekV2MoE.
        quant_format:     "nvfp4", "fp8", or "int8".
        device:           Compute device.
        nvfp4_block_size: Microscaling block size (16 for NVFP4 hardware).

    Returns:
        routed_gptq : dict mapping expert index e → (gate_gptq, up_gptq, down_gptq)
        shared_gptq : tuple (gate_gptq, up_gptq, down_gptq) for the shared expert
    """
    moe = layer.mlp
    routed_gptq: Dict[int, Tuple[GPTQ, GPTQ, GPTQ]] = {}

    for e, expert in enumerate(moe.experts):
        g_gate = GPTQ(expert.gate_proj)
        g_gate.quantizer = _make_quantizer(quant_format, device, nvfp4_block_size)

        g_up = GPTQ(expert.up_proj)
        g_up.quantizer = _make_quantizer(quant_format, device, nvfp4_block_size)

        g_down = GPTQ(expert.down_proj)
        g_down.quantizer = _make_quantizer(quant_format, device, nvfp4_block_size)

        routed_gptq[e] = (g_gate, g_up, g_down)

    # shared expert — same DeepseekV2MLP structure, all tokens flow through it
    se = moe.shared_experts
    shared_gptq: Tuple[GPTQ, GPTQ, GPTQ] = (
        GPTQ(se.gate_proj),
        GPTQ(se.up_proj),
        GPTQ(se.down_proj),
    )
    for g in shared_gptq:
        g.quantizer = _make_quantizer(quant_format, device, nvfp4_block_size)

    return routed_gptq, shared_gptq


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 2 — Phase 2: register hooks
# Call BEFORE the calibration forward passes. Remove handles AFTER.
# ─────────────────────────────────────────────────────────────────────────────

def register_expert_hooks(
    layer,
    routed_gptq: Dict[int, Tuple[GPTQ, GPTQ, GPTQ]],
    shared_gptq: Tuple[GPTQ, GPTQ, GPTQ],
) -> List:
    """Register forward hooks on each expert's Linear layers to collect Hessians.

    The MoE routing inside DeepseekV2MoE.forward determines which tokens each
    expert receives before calling the expert's Linear layers. A forward hook
    on nn.Linear receives exactly those token activations as inp[0], so no MoE
    forward patching is needed.

    Hessian inputs:
      gate_proj  hook: inp[0] = routed hidden states      [T_e, hidden]
      up_proj    hook: inp[0] = routed hidden states      [T_e, hidden]  (same)
      down_proj  hook: inp[0] = act_fn(gate) * up         [T_e, intermediate]

    Args:
        layer:        Transformer layer containing the MoE MLP.
        routed_gptq:  From setup_expert_gptq_instances.
        shared_gptq:  From setup_expert_gptq_instances.

    Returns:
        List of hook handles. Call handle.remove() on each after calibration.
    """
    handles = []

    def _make_hook(gptq_instance):
        def hook(module, inp, out):
            # inp[0] may be [batch, seq, hidden] or [T_e, hidden] depending on
            # how the MoE forward is implemented — flatten to [N, hidden].
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            gptq_instance.add_batch(x, None)
        return hook

    moe = layer.mlp

    # routed experts
    for e, expert in enumerate(moe.experts):
        g_gate, g_up, g_down = routed_gptq[e]
        handles.append(expert.gate_proj.register_forward_hook(_make_hook(g_gate)))
        handles.append(expert.up_proj.register_forward_hook(_make_hook(g_up)))
        handles.append(expert.down_proj.register_forward_hook(_make_hook(g_down)))

    # shared expert — all tokens flow through, no routing
    se = moe.shared_experts
    g_gate, g_up, g_down = shared_gptq
    handles.append(se.gate_proj.register_forward_hook(_make_hook(g_gate)))
    handles.append(se.up_proj.register_forward_hook(_make_hook(g_up)))
    handles.append(se.down_proj.register_forward_hook(_make_hook(g_down)))

    return handles


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 3 — Phase 3: quantize + write-back
# Call AFTER standard linear quantization, BEFORE layer.cpu().
# ─────────────────────────────────────────────────────────────────────────────

def quantize_and_writeback_experts(
    layer,
    routed_gptq: Dict[int, Tuple[GPTQ, GPTQ, GPTQ]],
    shared_gptq: Tuple[GPTQ, GPTQ, GPTQ],
    blocksize: int   = 128,
    percdamp: float  = 0.01,
    mixed_precision_threshold: float = float("inf"),
    device: str = "cuda",                              # ← add this
) -> Dict:
    """Run GPTQ on every expert linear and update weights in-place.

    Since experts are nn.Linear, fasterquant_blockwise updates layer.weight
    directly — no manual write-back is needed.

    Experts with zero calibration signal (nsamples == 0) fall back to
    round-to-nearest (RTN). With 512 calibration samples and top-6 routing
    over 64 experts, all experts should receive sufficient samples.

    Projections whose GPTQ loss exceeds mixed_precision_threshold are kept
    in BF16 (original weights restored).

    Args:
        layer:                     Transformer layer containing the MoE MLP.
        routed_gptq:               From setup_expert_gptq_instances.
        shared_gptq:               From setup_expert_gptq_instances.
        blocksize:                 GPTQ block width.
        percdamp:                  Hessian damping factor.
        mixed_precision_threshold: Max acceptable GPTQ loss; keep BF16 if exceeded.

    Returns:
        losses: dict with keys "routed" and "shared":
            losses["routed"][e][proj_name] → float loss, "RTN", or "BF16_kept"
            losses["shared"][proj_name]    → float loss, "RTN", or "BF16_kept"
    """
    losses   = {"routed": {}, "shared": {}}
    rtn_count  = 0
    kept_bf16  = 0
    proj_names = ["gate_proj", "up_proj", "down_proj"]

    # ── routed experts ────────────────────────────────────────────────────────
    for e, (g_gate, g_up, g_down) in routed_gptq.items():
        losses["routed"][e] = {}
        for proj_name, g in zip(proj_names, (g_gate, g_up, g_down)):
            if g.nsamples == 0:
                _rtn_linear(g.layer, g.quantizer)
                losses["routed"][e][proj_name] = "RTN"
                rtn_count += 1
            else:
                orig_weight = g.layer.weight.data.clone()
                if g.H is not None:
                    g.H = g.H.to(device)
                loss = g.fasterquant_blockwise(blocksize=blocksize, percdamp=percdamp)
                if loss > mixed_precision_threshold:
                    g.layer.weight.data.copy_(orig_weight)
                    losses["routed"][e][proj_name] = "BF16_kept"
                    kept_bf16 += 1
                else:
                    losses["routed"][e][proj_name] = loss
            g.free()

    # ── shared expert ─────────────────────────────────────────────────────────
    for proj_name, g in zip(proj_names, shared_gptq):
        if g.nsamples == 0:
            _rtn_linear(g.layer, g.quantizer)
            losses["shared"][proj_name] = "RTN"
            rtn_count += 1
        else:
            orig_weight = g.layer.weight.data.clone()
            if g.H is not None:
                g.H = g.H.to(device)
            loss = g.fasterquant_blockwise(blocksize=blocksize, percdamp=percdamp)
            if loss > mixed_precision_threshold:
                g.layer.weight.data.copy_(orig_weight)
                losses["shared"][proj_name] = "BF16_kept"
                kept_bf16 += 1
            else:
                losses["shared"][proj_name] = loss
        g.free()

    if rtn_count:
        print(f"  [deepseek_v2] {rtn_count} projection(s) fell back to RTN "
              f"(zero calibration samples).")
    if kept_bf16:
        print(f"  [deepseek_v2] {kept_bf16} projection(s) kept BF16 "
              f"(loss > threshold {mixed_precision_threshold}).")

    return losses


def _rtn_linear(linear: nn.Linear, quantizer):
    """Round-to-nearest fallback for a standard nn.Linear [out, in] weight."""
    w   = linear.weight.data.float()
    quantizer.find_params(w)
    w_q = quantizer.quantize_dequantize(w)
    linear.weight.data.copy_(w_q.to(linear.weight.dtype))