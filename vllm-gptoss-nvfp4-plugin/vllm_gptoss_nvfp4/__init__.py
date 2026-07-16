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


def register():
    """vllm.general_plugins entry point — runs in every vLLM process."""
    try:
        _patch_create_weights()
        _patch_quant_config()
        _patch_gptoss_loader()
        logger.info("[gptoss-nvfp4] vLLM GPT-OSS NVFP4 patches applied")
    except Exception:                                     # noqa: BLE001
        logger.exception("[gptoss-nvfp4] failed to apply patches")
        raise
