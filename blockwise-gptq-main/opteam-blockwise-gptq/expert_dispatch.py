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

    Holds only the running Hessian and an nsamples counter. No weight matrix
    is stored — expert shims are created lazily one at a time during the
    quantization phase to avoid ~76 GB of simultaneous weight copies.

    P0.3 fix: accumulation delegates to gptq.accumulate_hessian(), the single
    canonical convention (H = (2/N)·Σ x xᵀ over flattened rows), so
    transferring (H, nsamples) into a real GPTQ instance before fasterquant
    is numerically identical to having called GPTQ.add_batch directly.
    The previous inline formula weighted each chunk by n/N² instead of the
    uniform per-row weight, systematically underweighting later calibration
    batches whenever expert token counts varied between forward passes.
    """

    def __init__(self, in_features: int):
        self.columns  = in_features
        self.H        = None
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out=None):
        from gptq import accumulate_hessian
        if self.H is None:
            self.H = torch.zeros(
                (self.columns, self.columns),
                device=inp.device,
                dtype=torch.float32,
            )
        self.H, self.nsamples = accumulate_hessian(self.H, self.nsamples, inp)


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
        capture_artifacts: bool = False,
    ) -> dict:
        """Quantize all expert weights using accumulated Hessians, write back in-place.

        Args:
            capture_artifacts: If True (nvfp4 only), record each slice's exact
                E2M1 codes and FP8 scales (P0.6) in the returned dict.

        Returns:
            losses — architecture-specific dict, passed to summarize_losses().
        """
        raise NotImplementedError

    def build_records(self, layer, expert_losses, *, layer_idx, prefix,
                      quant_format, blocksize, nvfp4_block_size, acc_state):
        """Build P0.5 manifest records from a capture-enabled quantize() result.

        Returns:
            (records, artifacts) — list of manifest record dicts (with
            "artifact" set to "pending" where an artifact exists) and
            {(name, expert_idx): QuantizedTensorArtifact}.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support artifact capture"
        )

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

    # ── optional overrides (implementation-consistent collection, P0.4) ───────

    def attach_passthrough(self, layer) -> Any:
        """Pin this layer's expert forward to the SAME implementation the
        Hessian-collection patch uses, WITHOUT collecting anything.

        Why: transformers >= 5.x dispatches MoE expert forwards through
        `use_experts_implementation` (`config._experts_implementation`, e.g.
        "batched_mm"), which is mathematically equivalent but ULP-different
        from the eager loop our collection patch reimplements. If only the
        group's layers were patched, downstream activations would depend on
        WHICH layers are in the group, making grouped collection
        non-reproducible across group sizes. Pinning every MoE layer to the
        loop implementation during every pass makes collection bitwise
        grouping-invariant.

        Default: no-op (correct for architectures whose expert handling does
        not replace the forward, e.g. DeepSeek V2 nn.Linear hooks).

        Returns:
            token to pass to detach_passthrough, or None.
        """
        return None

    def detach_passthrough(self, layer, token):
        """Undo attach_passthrough. Default: no-op."""
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
            # Incremented by the patched forward on every invocation; the
            # collection loop uses it to fail loudly if a fused kernel
            # forward bypassed the patch (P0.4 fail-closed check).
            "call_counter": {"n": 0},
        }

    def attach_hooks(self, layer, acc_state) -> Any:
        from gpt_oss_expert_gptq import patch_expert_forward
        original_forward = patch_expert_forward(
            layer.mlp.experts,
            acc_state["gu_h_map"],
            acc_state["dn_h_map"],
            call_counter=acc_state.get("call_counter"),
        )
        return original_forward          # hook_token = original forward callable

    def detach_hooks(self, layer, hook_token):
        layer.mlp.experts.forward = hook_token

    def attach_passthrough(self, layer) -> Any:
        """Pin the expert forward to the collection-patch loop implementation
        with no-op accumulators (see MoEHandler.attach_passthrough)."""
        from gpt_oss_expert_gptq import patch_expert_forward

        class _Null:
            def add_batch(self, inp, out=None):
                pass

        null = _Null()
        num_exp = layer.mlp.experts.gate_up_proj.shape[0]
        null_map = {e: null for e in range(num_exp)}
        return patch_expert_forward(layer.mlp.experts, null_map, dict(null_map))

    def detach_passthrough(self, layer, token):
        layer.mlp.experts.forward = token

    def quantize(
        self, layer, acc_state,
        quant_format, device, nvfp4_block_size,
        blocksize, percdamp, threshold,
        capture_artifacts: bool = False,
    ) -> dict:
        """Build expert shims ONE AT A TIME from _GptqH accumulators and quantize.

        This avoids holding all expert weight copies in memory simultaneously.
        If loss > threshold the shim's quantized weight is discarded and the
        original batched tensor is left untouched (BF16).  Cholesky failures
        fall back to BF16 for that expert only (recorded, never silent).

        With capture_artifacts=True (nvfp4 only), each quantized slice's exact
        E2M1 codes/FP8 scales are captured and verified bit-exact against the
        written-back weight (P0.6) before continuing.

        Returns:
            {"gu": {e: loss|"RTN"|"BF16"}, "dn": {...},
             "gu_reason"/"dn_reason": {e: str|None},
             "gu_art"/"dn_art": {e: QuantizedTensorArtifact|None}}   (capture only)
        """
        from gpt_oss_expert_gptq import (
            _make_quantizer as _exp_quant,
            _rtn_inplace,
        )
        from quant_artifacts import verify_artifact_matches

        experts      = layer.mlp.experts
        orig_dtype   = experts.gate_up_proj.dtype
        num_experts  = experts.gate_up_proj.shape[0]
        hidden       = experts.gate_up_proj.shape[1]
        gate_up_out  = experts.gate_up_proj.shape[2]
        intermediate = experts.down_proj.shape[1]

        gu_h_map = acc_state["gu_h_map"]
        dn_h_map = acc_state["dn_h_map"]
        result = {"gu": {}, "dn": {}, "gu_reason": {}, "dn_reason": {},
                  "gu_art": {}, "dn_art": {}}

        def _one_slice(side, e, h, batched, in_f, out_f):
            """Quantize one expert slice; fill result[side]/[side_reason]/[side_art].

            batched: the [in, out] slice view in the model (written in-place).
            The quantization operates on the transposed [out, in] orientation.
            """
            losses  = result[side]
            reasons = result[f"{side}_reason"]
            arts    = result[f"{side}_art"]
            label   = "gate_up " if side == "gu" else "down    "

            if h.nsamples == 0 or h.H is None:
                # RTN fallback for never-activated experts — explicit, recorded.
                q = _exp_quant(quant_format, device, nvfp4_block_size)
                if hasattr(q, "set_global_scale_from"):        # D-010, nvfp4
                    q.set_global_scale_from(batched)
                if capture_artifacts:
                    q.begin_capture(out_f, in_f)
                _rtn_inplace(batched, q)
                if capture_artifacts:
                    art = q.end_capture()
                    verify_artifact_matches(
                        art, batched.T,
                        what=f"expert[{e}].{side} (RTN)")
                    arts[e] = art
                losses[e] = "RTN"
                reasons[e] = "no calibration samples reached this expert"
                print(f"    expert[{e:02d}] {label} → RTN  (no Hessian)")
                return

            shim = nn.Linear(in_f, out_f, bias=False, dtype=torch.float32)
            shim.weight.data.copy_(batched.detach().T.float())
            shim = shim.to(device)
            g = GPTQ(shim)
            g.quantizer = _exp_quant(quant_format, device, nvfp4_block_size)
            if hasattr(g.quantizer, "set_global_scale_from"):   # D-010, nvfp4
                g.quantizer.set_global_scale_from(shim.weight)
            g.H        = h.H.to(device)
            g.nsamples = h.nsamples
            if capture_artifacts:
                g.quantizer.begin_capture(out_f, in_f)
            try:
                loss = g.fasterquant_blockwise(blocksize=blocksize,
                                               percdamp=percdamp)
                if math.isnan(loss):
                    raise ValueError("fasterquant returned NaN loss")
                if threshold is not None and loss > threshold:
                    if capture_artifacts:
                        g.quantizer.abort_capture()
                    losses[e] = "BF16"
                    reasons[e] = f"loss {loss:.4f} > threshold {threshold}"
                    print(f"    expert[{e:02d}] {label} → BF16 "
                          f"(loss={loss:.2f} > {threshold})")
                else:
                    if capture_artifacts:
                        art = g.quantizer.end_capture()
                        # P0.6: verify against the fp32 shim BEFORE the bf16
                        # writeback cast — strictest possible comparison.
                        verify_artifact_matches(
                            art, g.layer.weight.data,
                            what=f"expert[{e}].{side}")
                        arts[e] = art
                    batched.copy_(g.layer.weight.data.T.to(orig_dtype))
                    losses[e] = loss
                    reasons[e] = None
                    print(f"    expert[{e:02d}] {label} → NVFP4 "
                          f"(loss={loss:.4f})")
            except Exception as exc:
                if capture_artifacts:
                    g.quantizer.abort_capture()
                warnings.warn(
                    f"[GPTQ] Expert {e} {side} failed "
                    f"({type(exc).__name__}: {exc}); keeping BF16."
                )
                losses[e] = "BF16"
                reasons[e] = f"exception: {type(exc).__name__}: {exc}"
                print(f"    expert[{e:02d}] {label} → BF16 "
                      f"(exception: {type(exc).__name__})")
            finally:
                g.free()
                del shim

        for e in range(num_experts):
            _one_slice("gu", e, gu_h_map[e], experts.gate_up_proj.data[e],
                       in_f=hidden, out_f=gate_up_out)
            _one_slice("dn", e, dn_h_map[e], experts.down_proj.data[e],
                       in_f=intermediate, out_f=hidden)

        if not capture_artifacts:
            result.pop("gu_art"); result.pop("dn_art")
        return result

    def build_records(self, layer, expert_losses, *, layer_idx, prefix,
                      quant_format, blocksize, nvfp4_block_size, acc_state):
        """Manifest records for every expert slice of this layer (P0.5)."""
        experts     = layer.mlp.experts
        num_experts = experts.gate_up_proj.shape[0]
        dtype_str   = str(experts.gate_up_proj.dtype)

        sides = {
            "gu": ("gate_up", f"{prefix}.mlp.experts.gate_up_proj",
                   experts.gate_up_proj.shape[2], experts.gate_up_proj.shape[1],
                   acc_state["gu_h_map"]),
            "dn": ("down", f"{prefix}.mlp.experts.down_proj",
                   experts.down_proj.shape[2], experts.down_proj.shape[1],
                   acc_state["dn_h_map"]),
        }

        records, artifacts = [], {}
        for side, (proj, name, out_f, in_f, h_map) in sides.items():
            losses  = expert_losses[side]
            reasons = expert_losses.get(f"{side}_reason", {})
            arts    = expert_losses.get(f"{side}_art", {})
            for e in range(num_experts):
                v = losses.get(e)
                if isinstance(v, float) and not math.isnan(v):
                    disposition, loss = "GPTQ_NVFP4", v
                elif v == "RTN":
                    disposition, loss = "RTN_NVFP4", None
                else:
                    disposition, loss = "BF16_FALLBACK", None
                art = arts.get(e)
                records.append({
                    "name": name,
                    "param": name,
                    "kind": "expert_slice",
                    "layer_index": layer_idx,
                    "projection": proj,
                    "expert_index": e,
                    "orig_shape": [out_f, in_f],
                    "orientation": "transposed_out_in",
                    "orig_dtype": dtype_str,
                    "requested_format": quant_format,
                    "disposition": disposition,
                    "reason": reasons.get(e),
                    "gptq_blocksize": blocksize,
                    "scale_block_size": nvfp4_block_size,
                    "loss": loss,
                    "normalized_loss": (loss / (out_f * in_f)
                                        if loss is not None else None),
                    "hessian_nsamples": h_map[e].nsamples,
                    "artifact": "pending" if art is not None else None,
                })
                if art is not None:
                    artifacts[(name, e)] = art
        return records, artifacts

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
        capture_artifacts: bool = False,
    ) -> dict:
        if capture_artifacts:
            raise NotImplementedError(
                "Exact artifact capture (P0.6) is not implemented for the "
                "DeepSeek V2 expert path — run without artifact_dir."
            )
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