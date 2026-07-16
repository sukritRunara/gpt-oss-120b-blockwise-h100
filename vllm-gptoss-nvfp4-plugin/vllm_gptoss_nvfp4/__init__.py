"""vLLM 0.25.1 patches for GPT-OSS ModelOpt W4A16_NVFP4 checkpoints.

Loaded via the `vllm.general_plugins` entry point in EVERY vLLM process
(including the V1 engine-core subprocess). Three surgical patches — see
docs/VLLM_NVFP4_CONTRACT.md §6 in the project repo for the full analysis:

P1  ModelOptNvFp4FusedMoE.create_weights does not register w13_bias/w2_bias
    even when the MoE has biases (GPT-OSS does). Without them, loading a
    checkpoint that contains expert biases dies with
    KeyError 'layers.N.mlp.experts.w2_bias'. Mirror of the mxfp4 method's
    bias registration (vllm/model_executor/layers/quantization/mxfp4.py:254).

P2  ModelOptNvFp4FusedMoE.get_fused_moe_quant_config (Marlin backend) builds
    a quant config without biases and without the swigluoai activation
    constants, so gpt-oss experts would silently run plain SiLU with no bias.
    Rebuild the config with w1_bias/w2_bias and gemm1_alpha=1.702,
    gemm1_beta=1.0, gemm1_clamp_limit=7.0 — the same constants vLLM's own
    mxfp4 method hardcodes for gpt-oss (mxfp4.py:416-418).

P3  GptOssModel._load_weights_other branches on the SUBSTRING
    ".w13_weight"/" .w2_weight" (gpt_oss.py:1020,1035), so the 2-D
    *_weight_scale_2 tensors of a ModelOpt MoE checkpoint would enter the
    3-D narrow+permute path and crash. Route them (and *_input_scale, which
    the generic path would handle but we keep symmetric) directly into their
    parameters before the branchy loop. TP=1 / EP=1 only — asserted.

P4  prepare_nvfp4_moe_layer_for_marlin pads the per-shard intermediate size
    to Marlin tiles (2880 → 2944 for gpt-oss) and marlin-permutes weights
    and scales — but never touches the biases (the mxfp4 marlin prep does:
    marlin_utils_fp4.prepare_moe_mxfp4_layer_for_marlin permute_bias). The
    Marlin MoE kernel then asserts `b_bias.size(1) != size_n` and aborts.
    After the stock process_weights_after_loading, convert the biases to
    kernel format: pad w13_bias per gate/up shard to padded_N (mirroring
    pad_w13's row layout) and marlin_permute_bias both biases per expert.

All patches are no-ops for non-GPT-OSS models and non-NVFP4 checkpoints.
"""

import logging

logger = logging.getLogger(__name__)

_GPTOSS_SWIGLU_ALPHA = 1.702
_GPTOSS_SWIGLU_BETA = 1.0
_GPTOSS_SWIGLU_LIMIT = 7.0


def _patch_create_weights():
    import torch
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.utils import set_weight_attrs

    orig = ModelOptNvFp4FusedMoE.create_weights

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        orig(self, layer, num_experts, hidden_size,
             intermediate_size_per_partition, params_dtype,
             **extra_weight_attrs)
        # P1: expert biases (GPT-OSS has_bias MoE). Same registration the
        # mxfp4 method performs; zero-init, overwritten by the checkpoint.
        if getattr(self.moe, "has_bias", False) \
                and not hasattr(layer, "w13_bias"):
            shards = 2 if self.moe.is_act_and_mul else 1
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts,
                            shards * intermediate_size_per_partition,
                            dtype=torch.bfloat16),
                requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=torch.bfloat16),
                requires_grad=False)
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
            logger.info("[gptoss-nvfp4] registered w13_bias/w2_bias on %s",
                        getattr(layer, "layer_name", type(layer).__name__))

    ModelOptNvFp4FusedMoE.create_weights = create_weights


def _patch_quant_config():
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
    )

    orig = ModelOptNvFp4FusedMoE.get_fused_moe_quant_config

    def get_fused_moe_quant_config(self, layer):
        w13_bias = getattr(layer, "w13_bias", None)
        if self.nvfp4_backend == NvFp4MoeBackend.MARLIN \
                and w13_bias is not None:
            # P2: same tensors the stock Marlin branch passes
            # (make_nvfp4_moe_quant_config → nvfp4_w4a16_moe_quant_config),
            # plus biases and the gpt-oss swigluoai constants.
            return FusedMoEQuantConfig.make(
                quant_dtype=None,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                g1_alphas=layer.w13_weight_scale_2,
                g2_alphas=layer.w2_weight_scale_2,
                weight_dtype="nvfp4",
                gemm1_alpha=_GPTOSS_SWIGLU_ALPHA,
                gemm1_beta=_GPTOSS_SWIGLU_BETA,
                gemm1_clamp_limit=_GPTOSS_SWIGLU_LIMIT,
                w1_bias=w13_bias,
                w2_bias=getattr(layer, "w2_bias", None),
            )
        return orig(self, layer)

    ModelOptNvFp4FusedMoE.get_fused_moe_quant_config = get_fused_moe_quant_config


def _patch_gptoss_loader():
    from vllm.model_executor.model_loader.weight_utils import (
        maybe_remap_moe_expert_param_name,
    )
    from vllm.model_executor.models.gpt_oss import GptOssModel

    orig = GptOssModel._load_weights_other

    _SCALE2_SUFFIXES = ("w13_weight_scale_2", "w2_weight_scale_2",
                        "w13_input_scale", "w2_input_scale")

    def _load_weights_other(self, ep_rank_end, ep_rank_start, heads_per_rank,
                            head_start, weights, stacked_params_mapping):
        params_dict = dict(self.named_parameters())
        extra_loaded = set()

        def filtered():
            for name, w in weights:
                if name.endswith(_SCALE2_SUFFIXES):
                    # P3: keep 2-D per-expert scales away from the 3-D
                    # narrow+permute branch. Direct copy is exact at TP1/EP1.
                    rname = maybe_remap_moe_expert_param_name(name, params_dict)
                    param = params_dict.get(rname)
                    if param is None:
                        logger.warning("[gptoss-nvfp4] no param for %s; "
                                       "skipping", rname)
                        continue
                    if param.shape != w.shape:
                        raise ValueError(
                            f"[gptoss-nvfp4] {rname}: checkpoint shape "
                            f"{tuple(w.shape)} != param {tuple(param.shape)} "
                            f"(this patch supports TP=1/EP=1 only)")
                    param.data.copy_(w)
                    extra_loaded.add(rname)
                    continue
                yield name, w

        loaded = orig(self, ep_rank_end, ep_rank_start, heads_per_rank,
                      head_start, filtered(), stacked_params_mapping)
        return loaded | extra_loaded

    GptOssModel._load_weights_other = _load_weights_other


def _patch_bias_kernel_format():
    import torch
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_bias,
    )

    orig = ModelOptNvFp4FusedMoE.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        # P4/P5: run BEFORE the stock method — it builds the quant config and
        # the modular kernel at its end (modelopt.py:1600+), so the layer must
        # already be in a Marlin-consistent layout.
        #
        # Marlin's NVFP4 MoE prep (prepare_nvfp4_moe_layer_for_marlin) pads
        # each gate/up shard to Marlin tiles by VIEWING w13 as
        # (E, 2, N, cols) — i.e. it assumes gate rows 0..N-1 and up rows
        # N..2N-1 (CONCATENATED). GPT-OSS checkpoints (and the HF/vLLM BF16
        # path) INTERLEAVE gate/up rows. With gpt-oss dims (N=2880, K=2880:
        # K%128!=0 → padded_N=2944) the interleaved rows get scrambled by the
        # shard-view pad and the activation misreads pairs → garbage output.
        # Fix: de-interleave w13 rows (weights, scales, bias) to the
        # concatenated layout and switch the activation to
        # SWIGLUOAI_UNINTERLEAVE (identical math on concatenated halves).
        # Additionally pad + marlin-permute the biases, which the NVFP4 prep
        # never handles (the mxfp4 prep does — marlin_utils_fp4.py
        # permute_bias) — the kernel otherwise asserts b_bias.size(1)!=size_n.
        if (getattr(layer, "w13_bias", None) is not None
                and self.nvfp4_backend == NvFp4MoeBackend.MARLIN):
            import os

            from vllm.model_executor.layers.fused_moe.activation import (
                MoEActivation,
            )

            # Debug knobs for contract bisection (default = current best
            # understanding; see docs/VLLM_NVFP4_CONTRACT.md §6).
            deint_on = os.environ.get("GPTOSS_NVFP4_DEINT", "1") == "1"
            bias_permute_on = os.environ.get(
                "GPTOSS_NVFP4_BIAS_PERMUTE", "1") == "1"

            E = layer.num_experts
            N = layer.intermediate_size_per_partition
            K = layer.hidden_size

            w13_bias = layer.w13_bias.data
            w2_bias = layer.w2_bias.data

            if deint_on:
                # interleaved (g0,u0,g1,u1,…) → concatenated ([g…; u…])
                deint = torch.cat([torch.arange(0, 2 * N, 2),
                                   torch.arange(1, 2 * N, 2)])
                for pname in ("w13_weight", "w13_weight_scale"):
                    p = getattr(layer, pname)
                    idx = deint.to(p.data.device)
                    p.data.copy_(p.data.index_select(1, idx))
                w13_bias = w13_bias.index_select(
                    1, deint.to(w13_bias.device))
                layer.activation = MoEActivation.SWIGLUOAI_UNINTERLEAVE

            def round_up(x, m):
                return (x + m - 1) // m * m

            padded_N = round_up(N, 64) if K % 128 == 0 else round_up(N, 128)
            if padded_N != N:
                b = w13_bias.view(E, 2, N)
                b = torch.nn.functional.pad(b, (0, padded_N - N))
                w13_bias = b.reshape(E, 2 * padded_N)
            if bias_permute_on:
                w13_bias = torch.stack(
                    [marlin_permute_bias(w13_bias[e]) for e in range(E)])
                w2_bias = torch.stack(
                    [marlin_permute_bias(w2_bias[e]) for e in range(E)])

            layer.w13_bias = torch.nn.Parameter(w13_bias.contiguous(),
                                                requires_grad=False)
            layer.w2_bias = torch.nn.Parameter(w2_bias.contiguous(),
                                               requires_grad=False)
            logger.info("[gptoss-nvfp4] MoE prep: deint=%s bias_permute=%s "
                        "N %d → %d", deint_on, bias_permute_on, N, padded_N)

        orig(self, layer)

    ModelOptNvFp4FusedMoE.process_weights_after_loading = \
        process_weights_after_loading


def _patch_topk_weight_multiply():
    """P5: bypass the broken mul_topk_weights path in moe_wna16_marlin_gemm.

    Isolated with a standalone capture/replay harness against an exact
    reference: the SAME weights and schedule produce correct output with
    mul_topk_weights=False and fully corrupt output (~1e33, consistent with
    an out-of-bounds fp32 multiplier read) with mul_topk_weights=True at
    gpt-oss's shapes — across every thread config, block size, and reduce
    mode. Small shapes are unaffected (allocation-layout-dependent OOB).

    Workaround: call the gemm with mul_topk_weights=False and apply the
    routing weights as an elementwise multiply on the [M*topk, K] gemm2
    output — mathematically identical, negligible cost.

    Scope guard: only rewrites calls where the stock wrapper would have set
    mul_topk_weights=True for gemm2 (apply_router_weight_on_input=False,
    the gpt-oss configuration). The gemm1 mul path
    (apply_router_weight_on_input=True) is left untouched.
    """
    import torch
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm

    orig_gemm = mm.ops.moe_wna16_marlin_gemm
    orig_fused = mm._fused_marlin_moe

    def _fused_marlin_moe(hidden_states, w1, w2, bias1, bias2, w1_scale,
                          w2_scale, topk_weights, num_topk, quant_type,
                          apply_router_weight_on_input, expert_map,
                          block_size_m, sorted_token_ids, expert_ids,
                          num_tokens_post_padded, activation=None, **kw):
        if apply_router_weight_on_input:
            # gemm1 would carry the multiply — configuration not used by
            # gpt-oss; leave stock behavior.
            return orig_fused(hidden_states, w1, w2, bias1, bias2, w1_scale,
                              w2_scale, topk_weights, num_topk, quant_type,
                              apply_router_weight_on_input, expert_map,
                              block_size_m, sorted_token_ids, expert_ids,
                              num_tokens_post_padded, activation=activation,
                              **kw)

        # Intercept the second gemm (top_k == 1 in the stock wrapper) and
        # strip the in-kernel multiply; everything else passes through.
        def gemm_shim(a, c, b_q_weight, b_bias, b_scales, a_scales,
                      global_scale, b_zeros, g_idx, perm, workspace,
                      sorted_ids, expert_ids_, num_tokens_past_padded_,
                      topk_weights_, moe_block_size, top_k,
                      mul_topk_weights, b_q_type, size_m, size_n, size_k,
                      is_k_full, use_atomic_add, use_fp32_reduce,
                      is_zp_float):
            do_external_mul = mul_topk_weights and top_k == 1
            out = orig_gemm(a, c, b_q_weight, b_bias, b_scales, a_scales,
                            global_scale, b_zeros, g_idx, perm, workspace,
                            sorted_ids, expert_ids_, num_tokens_past_padded_,
                            topk_weights_, moe_block_size=moe_block_size,
                            top_k=top_k,
                            mul_topk_weights=(False if do_external_mul
                                              else mul_topk_weights),
                            b_q_type=b_q_type, size_m=size_m, size_n=size_n,
                            size_k=size_k, is_k_full=is_k_full,
                            use_atomic_add=use_atomic_add,
                            use_fp32_reduce=use_fp32_reduce,
                            is_zp_float=is_zp_float)
            if do_external_mul:
                # rows are per (token, topk-slot); weights broadcast per row
                out.mul_(topk_weights_.reshape(-1, 1).to(out.dtype))
            return out

        mm.ops.moe_wna16_marlin_gemm = gemm_shim
        try:
            return orig_fused(hidden_states, w1, w2, bias1, bias2, w1_scale,
                              w2_scale, topk_weights, num_topk, quant_type,
                              apply_router_weight_on_input, expert_map,
                              block_size_m, sorted_token_ids, expert_ids,
                              num_tokens_post_padded, activation=activation,
                              **kw)
        finally:
            mm.ops.moe_wna16_marlin_gemm = orig_gemm

    mm._fused_marlin_moe = _fused_marlin_moe


def register():
    """vllm.general_plugins entry point — runs in every vLLM process."""
    try:
        _patch_create_weights()
        _patch_quant_config()
        _patch_gptoss_loader()
        _patch_bias_kernel_format()
        _patch_topk_weight_multiply()
        logger.info("[gptoss-nvfp4] vLLM GPT-OSS NVFP4 patches applied")
    except Exception:                                     # noqa: BLE001
        logger.exception("[gptoss-nvfp4] failed to apply patches")
        raise
