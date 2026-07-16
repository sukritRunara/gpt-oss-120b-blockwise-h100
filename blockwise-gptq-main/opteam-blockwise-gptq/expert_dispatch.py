"""
MoE expert handler dispatch for GPTQ.

Provides a unified interface for model-specific expert extraction and
quantization. Each architecture that stores expert weights in a non-standard
way has a handler class that wraps its model-specific functions behind a
common interface that apply.py can call without knowing the underlying arch.

Supported architectures
-----------------------
    gpt_oss      — batched weight tensors (gate_up_proj[E, in, out]),
                   requires shim construction, transpose on write-back.
                   Uses lightweight _GptqH accumulators in parallel mode.

    deepseek_v2  — standard nn.Linear experts inside DeepseekV2MLP.
                   GPTQ wraps the Linear modules directly; forward hooks
                   collect Hessians without any MoE patching.

Adding a new architecture
-------------------------
    1. Write a model-specific expert_gptq module (e.g. my_model_expert_gptq.py).
    2. Implement a subclass of MoEHandler.
    3. Register it in _HANDLER_REGISTRY at the bottom of this file.

Note on _GptqH
--------------
    _GptqH (lightweight Hessian accumulator) lives here rather than in
    apply.py because it is a GPT-OSS implementation detail used exclusively
    by GptOssHandler.  apply.py no longer needs to know about it.
"""

import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from gptq import GPTQ


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Hessian accumulator  (used by GPT-OSS parallel mode)
# ─────────────────────────────────────────────────────────────────────────────

class _GptqH:
    """Hessian-only accumulator for the parallel collection phase.

    Holds only H = running X.T @ X average and an nsamples counter.
    No weight matrix is stored — expert shims are created lazily one at a time
    during the quantization phase to avoid ~76 GB of simultaneous weight copies.

    The add_batch accumulation formula is identical to GPTQ.add_batch so that
    transferring (H, nsamples) into a real GPTQ instance before fasterquant
    yields numerically correct results.
    """

    def __init__(self, in_features: int):
        self.columns  = in_features
        self.H        = None
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out=None):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float()
        n   = inp.shape[0]
        if self.H is None:
            self.H = torch.zeros(
                (self.columns, self.columns),
                device=inp.device,
                dtype=torch.float32,
            )
        self.nsamples += n
        self.H        *= (self.nsamples - n) / self.nsamples
        inp            = inp / math.sqrt(self.nsamples)
        self.H        += inp.T @ inp * (n / self.nsamples)


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class MoEHandler:
    """Base class for architecture-specific MoE expert handlers.

    Every method operates on a **single** transformer layer.  apply.py calls
    these methods inside its per-layer loops without knowing which handler is
    active.

    Life-cycle inside apply.py (parallel mode)
    ------------------------------------------
        # Phase 1 — setup (before forward pass)
        acc_state  = handler.setup_accumulators(layer, device, ...)
        hook_token = handler.attach_hooks(layer, acc_state)

        # Phase 1 — after forward pass
        handler.detach_hooks(layer, hook_token)

        # (optional) Hessian caching
        payload = handler.hessian_state_to_save(acc_state)
        handler.load_hessian_state(acc_state, payload)

        # Phase 2 — quantize
        expert_losses = handler.quantize(layer, acc_state, ...)

        # Cleanup
        handler.free_hessians(acc_state)

    Life-cycle inside apply.py (sequential mode)
    ---------------------------------------------
        # Per-layer, before calibration forward passes
        acc_state  = handler.setup_accumulators(layer, device, ...)
        hook_token = handler.attach_hooks(layer, acc_state)
        # ... calibration forward passes ...
        handler.detach_hooks(layer, hook_token)

        # Quantize immediately (no caching needed)
        expert_losses = handler.quantize(layer, acc_state, ...)
    """

    # ── required overrides ─────────────────────────────────────────────────────

    def has_moe(self, layer) -> bool:
        """Return True if *this layer* has MoE experts requiring special handling."""
        raise NotImplementedError

    def num_experts(self, layer) -> int:
        """Return the number of *routed* experts in this layer (for summary counts)."""
        raise NotImplementedError

    def setup_accumulators(
        self, layer, device, quant_format=None, nvfp4_block_size: int = 16
    ) -> Any:
        """Create Hessian accumulators (or full GPTQ instances) for all experts.

        Args:
            layer:            Transformer layer.
            device:           Compute device.
            quant_format:     Required by DeepSeek V2 (GPTQ instances need it);
                              ignored by GPT-OSS (_GptqH has no quantizer).
            nvfp4_block_size: Block size for NVFP4 quantizers.

        Returns:
            acc_state — architecture-specific dict passed to all other methods.
        """
        raise NotImplementedError

    def attach_hooks(self, layer, acc_state) -> Any:
        """Register forward hooks / patch the expert forward to collect Hessians.

        Returns:
            hook_token — what is needed to undo the hooks:
                GPT-OSS:     original_forward callable  (restore via .forward =)
                DeepSeek V2: list of hook handles        (remove via .remove())
        """
        raise NotImplementedError

    def detach_hooks(self, layer, hook_token):
        """Undo attach_hooks: remove hook handles or restore patched forward."""
        raise NotImplementedError

    def quantize(
        self, layer, acc_state,
        quant_format: str, device,
        nvfp4_block_size: int,
        blocksize: int, percdamp: float, threshold,
    ) -> dict:
        """Quantize all expert weights using accumulated Hessians, write back in-place.

        Returns:
            losses — architecture-specific dict, passed to summarize_losses().
        """
        raise NotImplementedError

    def summarize_losses(
        self, losses: dict
    ) -> Tuple[int, int, int, float, float]:
        """Parse a losses dict into a five-tuple for the summary table.

        Returns:
            (n_quantized, n_bf16, n_rtn, avg_loss_a, avg_loss_b)

            For GPT-OSS: avg_loss_a = gate_up average, avg_loss_b = down average.
            For DeepSeek V2: avg_loss_a == avg_loss_b = combined average.
        """
        raise NotImplementedError

    # ── optional overrides (Hessian caching) ──────────────────────────────────

    def hessian_state_to_save(self, acc_state) -> dict:
        """Return a serialisable dict of {H, nsamples} for all accumulators."""
        return {}

    def load_hessian_state(self, acc_state, payload: dict):
        """Restore {H, nsamples} from a saved payload into acc_state in-place."""
        pass

    def free_hessians(self, acc_state):
        """Set H=None on all accumulators to release GPU/CPU RAM after quantisation."""
        pass

    # ── optional override (layer filtering) ───────────────────────────────────

    def filter_standard_layers(self, layer, subset: dict) -> dict:
        """Remove expert layers from the find_layers() result.

        Prevents expert nn.Linear modules from being processed by the standard
        GPTQ loop in addition to the expert handler (double-processing).

        Default: return subset unchanged — correct for architectures where
        find_layers() cannot reach expert weights (e.g. GPT-OSS with batched
        tensors that are not nn.Linear).
        """
        return subset


# ─────────────────────────────────────────────────────────────────────────────
# GPT-OSS handler
# ─────────────────────────────────────────────────────────────────────────────

class GptOssHandler(MoEHandler):
    """Handler for GPT-OSS-style MoE expert quantization.

    Expert weights are stored as batched tensors:
        layer.mlp.experts.gate_up_proj  [E, in,  gate_up_out]
        layer.mlp.experts.down_proj     [E, in,  hidden]     (note: in=intermediate)

    These are NOT nn.Linear, so find_layers() cannot find them — no filtering
    is needed.

    Parallel mode uses lightweight _GptqH accumulators collected via a patched
    expert forward.  During quantization, shims are built one expert at a time
    to avoid holding all 32 × 2 weight copies simultaneously (~76 GB).
    """

    def has_moe(self, layer) -> bool:
        from gpt_oss_expert_gptq import is_gpt_oss_experts
        return is_gpt_oss_experts(layer)

    def num_experts(self, layer) -> int:
        return layer.mlp.experts.gate_up_proj.shape[0]

    # filter_standard_layers: no-op (base class default is correct)

    def setup_accumulators(
        self, layer, device, quant_format=None, nvfp4_block_size: int = 16
    ) -> dict:
        experts      = layer.mlp.experts
        hidden       = experts.gate_up_proj.shape[1]
        intermediate = experts.down_proj.shape[1]
        num_exp      = experts.gate_up_proj.shape[0]
        return {
            "gu_h_map": {e: _GptqH(hidden)       for e in range(num_exp)},
            "dn_h_map": {e: _GptqH(intermediate) for e in range(num_exp)},
        }

    def attach_hooks(self, layer, acc_state) -> Any:
        from gpt_oss_expert_gptq import patch_expert_forward
        original_forward = patch_expert_forward(
            layer.mlp.experts,
            acc_state["gu_h_map"],
            acc_state["dn_h_map"],
        )
        return original_forward          # hook_token = original forward callable

    def detach_hooks(self, layer, hook_token):
        layer.mlp.experts.forward = hook_token

    def quantize(
        self, layer, acc_state,
        quant_format, device, nvfp4_block_size,
        blocksize, percdamp, threshold,
    ) -> dict:
        """Build expert shims ONE AT A TIME from _GptqH accumulators and quantize.

        This avoids holding all expert weight copies in memory simultaneously.
        If loss > threshold the shim's quantized weight is discarded and the
        original batched tensor is left untouched (BF16).  Cholesky failures
        fall back to RTN for that expert only.
        """
        from gpt_oss_expert_gptq import (
            _make_quantizer as _exp_quant,
            _rtn_inplace,
        )

        experts      = layer.mlp.experts
        orig_dtype   = experts.gate_up_proj.dtype
        num_experts  = experts.gate_up_proj.shape[0]
        hidden       = experts.gate_up_proj.shape[1]
        gate_up_out  = experts.gate_up_proj.shape[2]
        intermediate = experts.down_proj.shape[1]

        gu_h_map  = acc_state["gu_h_map"]
        dn_h_map  = acc_state["dn_h_map"]
        gu_losses: Dict[int, Any] = {}
        dn_losses: Dict[int, Any] = {}

        for e in range(num_experts):
            h_gu = gu_h_map[e]
            h_dn = dn_h_map[e]

            # ── gate_up ───────────────────────────────────────────────────────
            if h_gu.nsamples == 0 or h_gu.H is None:
                q = _exp_quant(quant_format, device, nvfp4_block_size)
                q.find_params(experts.gate_up_proj.data[e].T.float())
                _rtn_inplace(experts.gate_up_proj.data[e], q)
                gu_losses[e] = "RTN"
                print(f"    expert[{e:02d}] gate_up  → RTN  (no Hessian)")
            else:
                shim = nn.Linear(hidden, gate_up_out, bias=False,
                                 dtype=torch.float32)
                shim.weight.data.copy_(
                    experts.gate_up_proj[e].detach().T.float()
                )
                shim = shim.to(device)
                g = GPTQ(shim)
                g.quantizer = _exp_quant(quant_format, device, nvfp4_block_size)
                g.H        = h_gu.H.to(device)
                g.nsamples = h_gu.nsamples
                try:
                    loss = g.fasterquant_blockwise(blocksize=blocksize,
                                                   percdamp=percdamp)
                    if math.isnan(loss):
                        raise ValueError("fasterquant returned NaN loss")
                    if threshold is not None and loss > threshold:
                        gu_losses[e] = "BF16"
                        print(f"    expert[{e:02d}] gate_up  → BF16 "
                              f"(loss={loss:.2f} > {threshold})")
                    else:
                        experts.gate_up_proj.data[e].copy_(
                            g.layer.weight.data.T.to(orig_dtype)
                        )
                        gu_losses[e] = loss
                        print(f"    expert[{e:02d}] gate_up  → NVFP4 "
                              f"(loss={loss:.4f})")
                except Exception as exc:
                    warnings.warn(
                        f"[GPTQ] Expert {e} gate_up failed "
                        f"({type(exc).__name__}: {exc}); keeping BF16."
                    )
                    gu_losses[e] = "BF16"
                    print(f"    expert[{e:02d}] gate_up  → BF16 "
                          f"(exception: {type(exc).__name__})")
                g.free()
                del shim

            # ── down_proj ─────────────────────────────────────────────────────
            if h_dn.nsamples == 0 or h_dn.H is None:
                q = _exp_quant(quant_format, device, nvfp4_block_size)
                q.find_params(experts.down_proj.data[e].T.float())
                _rtn_inplace(experts.down_proj.data[e], q)
                dn_losses[e] = "RTN"
                print(f"    expert[{e:02d}] down     → RTN  (no Hessian)")
            else:
                shim = nn.Linear(intermediate, hidden, bias=False,
                                 dtype=torch.float32)
                shim.weight.data.copy_(
                    experts.down_proj[e].detach().T.float()
                )
                shim = shim.to(device)
                g = GPTQ(shim)
                g.quantizer = _exp_quant(quant_format, device, nvfp4_block_size)
                g.H        = h_dn.H.to(device)
                g.nsamples = h_dn.nsamples
                try:
                    loss = g.fasterquant_blockwise(blocksize=blocksize,
                                                   percdamp=percdamp)
                    if math.isnan(loss):
                        raise ValueError("fasterquant returned NaN loss")
                    if threshold is not None and loss > threshold:
                        dn_losses[e] = "BF16"
                        print(f"    expert[{e:02d}] down     → BF16 "
                              f"(loss={loss:.2f} > {threshold})")
                    else:
                        experts.down_proj.data[e].copy_(
                            g.layer.weight.data.T.to(orig_dtype)
                        )
                        dn_losses[e] = loss
                        print(f"    expert[{e:02d}] down     → NVFP4 "
                              f"(loss={loss:.4f})")
                except Exception as exc:
                    warnings.warn(
                        f"[GPTQ] Expert {e} down failed "
                        f"({type(exc).__name__}: {exc}); keeping BF16."
                    )
                    dn_losses[e] = "BF16"
                    print(f"    expert[{e:02d}] down     → BF16 "
                          f"(exception: {type(exc).__name__})")
                g.free()
                del shim

        return {"gu": gu_losses, "dn": dn_losses}

    def summarize_losses(self, losses) -> Tuple[int, int, int, float, float]:
        def _v(x): return isinstance(x, float) and not math.isnan(x) and not math.isinf(x)
        gu, dn = losses["gu"], losses["dn"]
        n_gu   = sum(1 for v in gu.values() if _v(v))
        n_dn   = sum(1 for v in dn.values() if _v(v))
        n_bf16 = sum(1 for v in gu.values() if v == "BF16") + \
                 sum(1 for v in dn.values() if v == "BF16")
        n_rtn  = sum(1 for v in gu.values() if v == "RTN")
        avg_gu = sum(v for v in gu.values() if _v(v)) / n_gu if n_gu > 0 else 0.0
        avg_dn = sum(v for v in dn.values() if _v(v)) / n_dn if n_dn > 0 else 0.0
        return n_gu, n_bf16, n_rtn, avg_gu, avg_dn

    # ── Hessian caching ────────────────────────────────────────────────────────

    def hessian_state_to_save(self, acc_state) -> dict:
        payload = {"gu": {}, "dn": {}}
        for e, h in acc_state["gu_h_map"].items():
            if h.H is not None:
                payload["gu"][e] = {"H": h.H.cpu(), "nsamples": h.nsamples}
        for e, h in acc_state["dn_h_map"].items():
            if h.H is not None:
                payload["dn"][e] = {"H": h.H.cpu(), "nsamples": h.nsamples}
        return payload

    def load_hessian_state(self, acc_state, payload: dict):
        for e, h in acc_state["gu_h_map"].items():
            if e in payload.get("gu", {}):
                h.H        = payload["gu"][e]["H"]
                h.nsamples = payload["gu"][e]["nsamples"]
        for e, h in acc_state["dn_h_map"].items():
            if e in payload.get("dn", {}):
                h.H        = payload["dn"][e]["H"]
                h.nsamples = payload["dn"][e]["nsamples"]

    def free_hessians(self, acc_state):
        for h in acc_state["gu_h_map"].values():
            h.H = None
        for h in acc_state["dn_h_map"].values():
            h.H = None


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek V2 handler
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekV2Handler(MoEHandler):
    """Handler for DeepSeek V2 / V2-Lite MoE expert quantization.

    Experts are standard nn.Linear modules inside DeepseekV2MLP containers:
        layer.mlp.experts[e].gate_proj / up_proj / down_proj  — routed experts
        layer.mlp.shared_experts.gate_proj / up_proj / down_proj — shared expert

    Because experts are nn.Linear, find_layers() WILL find them.
    filter_standard_layers() removes them so the standard GPTQ loop does not
    process them a second time.

    GPTQ instances wrap the Linear modules directly; forward hooks on those
    modules collect Hessians — no MoE forward patching needed because routing
    happens above the Linear level, so each hook naturally receives only the
    token subset that expert actually processed.

    Layer 0 of DeepSeek V2 Lite is a dense MLP (no MoE) — has_moe() returns
    False for it and the standard GPTQ path handles it normally.
    """

    _EXPERT_PREFIXES = ("mlp.experts.", "mlp.shared_experts.")

    def has_moe(self, layer) -> bool:
        from deepseek_v2_lite_expert_gptq import is_deepseek_v2_moe
        return is_deepseek_v2_moe(layer)

    def num_experts(self, layer) -> int:
        return len(layer.mlp.experts)

    def filter_standard_layers(self, layer, subset: dict) -> dict:
        """Remove expert-owned linears; they're handled by DeepSeek expert path."""
        return {
            name: mod
            for name, mod in subset.items()
            if not any(name.startswith(p) for p in self._EXPERT_PREFIXES)
        }

    def setup_accumulators(
        self, layer, device, quant_format=None, nvfp4_block_size: int = 16
    ) -> dict:
        from deepseek_v2_lite_expert_gptq import setup_expert_gptq_instances
        routed_gptq, shared_gptq = setup_expert_gptq_instances(
            layer, quant_format, device, nvfp4_block_size
        )
        return {
            "routed_gptq":    routed_gptq,
            "shared_gptq":    shared_gptq,
            "quant_format":   quant_format,
            "nvfp4_block_size": nvfp4_block_size,
        }

    def attach_hooks(self, layer, acc_state) -> Any:
        from deepseek_v2_lite_expert_gptq import register_expert_hooks
        handles = register_expert_hooks(
            layer, acc_state["routed_gptq"], acc_state["shared_gptq"]
        )
        return handles   # hook_token = list of handles

    def detach_hooks(self, layer, hook_token):
        for h in hook_token:
            h.remove()

    def quantize(
        self, layer, acc_state,
        quant_format, device, nvfp4_block_size,
        blocksize, percdamp, threshold,
    ) -> dict:
        from deepseek_v2_lite_expert_gptq import quantize_and_writeback_experts
        return quantize_and_writeback_experts(
            layer,
            acc_state["routed_gptq"],
            acc_state["shared_gptq"],
            blocksize=blocksize,
            percdamp=percdamp,
            mixed_precision_threshold=(
                threshold if threshold is not None else float("inf")
            ),
        )

    def summarize_losses(self, losses) -> Tuple[int, int, int, float, float]:
        def _v(x): return isinstance(x, float) and not math.isnan(x) and not math.isinf(x)
        all_vals = []
        n_bf16 = 0; n_rtn = 0
        for e_losses in losses.get("routed", {}).values():
            for val in e_losses.values():
                if _v(val):        all_vals.append(val)
                elif val == "BF16_kept": n_bf16 += 1
                elif val == "RTN": n_rtn  += 1
        for val in losses.get("shared", {}).values():
            if _v(val):        all_vals.append(val)
            elif val == "BF16_kept": n_bf16 += 1
            elif val == "RTN": n_rtn  += 1
        n_q = len(all_vals)
        avg = sum(all_vals) / n_q if n_q > 0 else 0.0
        return n_q, n_bf16, n_rtn, avg, avg

    # ── Hessian caching ────────────────────────────────────────────────────────

    def hessian_state_to_save(self, acc_state) -> dict:
        _proj = ["gate_proj", "up_proj", "down_proj"]
        payload = {"routed": {}, "shared": {}}
        for e, (g_gate, g_up, g_dn) in acc_state["routed_gptq"].items():
            payload["routed"][e] = {}
            for name, g in zip(_proj, (g_gate, g_up, g_dn)):
                if g.H is not None:
                    payload["routed"][e][name] = {
                        "H": g.H.cpu(), "nsamples": g.nsamples
                    }
        g_gate, g_up, g_dn = acc_state["shared_gptq"]
        for name, g in zip(_proj, (g_gate, g_up, g_dn)):
            if g.H is not None:
                payload["shared"][name] = {"H": g.H.cpu(), "nsamples": g.nsamples}
        return payload

    def load_hessian_state(self, acc_state, payload: dict):
        _proj = ["gate_proj", "up_proj", "down_proj"]
        for e, (g_gate, g_up, g_dn) in acc_state["routed_gptq"].items():
            if e in payload.get("routed", {}):
                for name, g in zip(_proj, (g_gate, g_up, g_dn)):
                    if name in payload["routed"][e]:
                        g.H        = payload["routed"][e][name]["H"]
                        g.nsamples = payload["routed"][e][name]["nsamples"]
        g_gate, g_up, g_dn = acc_state["shared_gptq"]
        for name, g in zip(_proj, (g_gate, g_up, g_dn)):
            if name in payload.get("shared", {}):
                g.H        = payload["shared"][name]["H"]
                g.nsamples = payload["shared"][name]["nsamples"]

    def free_hessians(self, acc_state):
        for g_gate, g_up, g_dn in acc_state["routed_gptq"].values():
            for g in (g_gate, g_up, g_dn):
                g.H = None
        g_gate, g_up, g_dn = acc_state["shared_gptq"]
        for g in (g_gate, g_up, g_dn):
            g.H = None


# ─────────────────────────────────────────────────────────────────────────────
# Registry and factory
# ─────────────────────────────────────────────────────────────────────────────

_HANDLER_REGISTRY: Dict[str, MoEHandler] = {
    "gpt_oss":     GptOssHandler(),
    "deepseek_v2": DeepSeekV2Handler(),
}


def get_handler(arch_type: str) -> Optional[MoEHandler]:
    """Return the MoEHandler for arch_type, or None if no special handling is needed.

    Architectures without a handler (e.g. "llama", "opt", "qwen3_moe") are
    handled entirely by the standard find_layers() → GPTQ path in apply.py.

    Args:
        arch_type: Architecture string from model_utils.get_model_layers().

    Returns:
        MoEHandler instance, or None.
    """
    return _HANDLER_REGISTRY.get(arch_type, None)