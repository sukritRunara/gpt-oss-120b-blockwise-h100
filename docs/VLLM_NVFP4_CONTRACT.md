# vLLM NVFP4 W4A16 contract for GPT-OSS (pinned vLLM 0.25.1)

Verified against the **installed** `.venv-serve` source (vllm 0.25.1,
torch 2.11.0+cu130). All paths below are relative to
`.venv-serve/lib/python3.12/site-packages/vllm/`. Line numbers are from this
installation. **This is the authoritative packing contract for Stage 7.**

Verdict up front: **vLLM 0.25.1 supports W4A16_NVFP4 for BOTH the GPT-OSS
dense linears (FP4 Marlin GEMM) and the FusedMoE experts (Marlin MoE)** —
a full-NVFP4 GPT-OSS artifact is servable on H100. Two sharp edges exist
(§6): the `weight_scale_2` ModelOpt normalization convention, and a loader
bug for 2-D `*_weight_scale_2` MoE keys.

---

## 1. Config detection

`config.json` must carry a compressed-tensors-style `quantization_config`
(read by `ModelOptQuantConfigBase.from_config`,
`model_executor/layers/quantization/modelopt.py:281-368`):

```json
"quantization_config": {
  "quant_method": "modelopt",
  "quant_algo": "W4A16_NVFP4",
  "group_size": 16,
  "ignore": ["lm_head"]
}
```

- `quant_algo` and `group_size` are read **flat** from `quantization_config`
  (`modelopt.py:303-317`), NOT from `config_groups`.
- Exclusions come from the `ignore` list (`modelopt.py:316`,
  "'ignore' is the key in config.json") — wildcards matched against **vLLM
  module prefixes** (`is_layer_excluded`, `modelopt.py:144-160`), e.g.
  `model.layers.0.attn.qkv_proj` (note `attn`, not `self_attn`; q/k/v are
  fused).
- `"NVFP4" in quant_algo` promotes the method to `modelopt_fp4`
  (`override_quantization_method`, `modelopt.py:1063-1069`).
- `quant_method == "W4A16_NVFP4"` selects `ModelOptNvFp4W4A16LinearMethod`
  (`modelopt.py:1041-1046`) and `ModelOptNvFp4FusedMoE` in W4A16 mode
  (`use_a16`, `modelopt.py:1405-1416`).

## 2. Model classes and loader path

- Model: `GptOssForCausalLM` → `GptOssModel`
  (`model_executor/models/gpt_oss.py:1170`, `:295`).
- Experts: `MLPBlock` builds `FusedMoE(..., has_bias=True,
  activation="swigluoai", renormalize=True)` (`gpt_oss.py:214-227`).
- Router: `ReplicatedLinear` with `quant_config=None` (`gpt_oss.py:205-212`)
  — **never quantized**; keep `mlp.router.{weight,bias}` in BF16.
- `lm_head`: `ParallelLMHead` without quant_config (`gpt_oss.py:1219`) —
  keep BF16; keep `lm_head` in `ignore`.
- Checkpoint loading: `GptOssForCausalLM.load_weights` applies
  `hf_to_vllm_mapper` (`gpt_oss.py:1176-1203`), then
  `GptOssModel.load_weights` dispatches on
  `config.quantization_config["quant_method"]` (`gpt_oss.py:1121-1168`):
  `"modelopt"` → **`_load_weights_other`** (`gpt_oss.py:982`).
- Name mappings that matter (`gpt_oss.py:1176-1203`):
  `.self_attn.` → `.attn.`, `.embed_tokens.weight` → `.embedding.weight`,
  `.gate_up_proj_bias` → `.w13_bias`, `.down_proj_bias` → `.w2_bias`.
  We emit expert weight keys in vLLM-native `w13_*`/`w2_*` names directly
  (no suffix mapping fires on them).
- `swigluoai` activation is **interleaved** — gate = `x[..., ::2]`, up =
  `x[..., 1::2]` (`model_executor/layers/activation.py:434-452`), same as
  HF. Therefore w13 keeps the HF interleaved column order; no
  de-interleaving at pack time.

## 3. Dense linear tensors (attention q/k/v/o)

Created by `ModelOptNvFp4W4A16LinearMethod.create_weights`
(`modelopt.py:1282-1355`); checkpoint keys use HF names
(`...self_attn.q_proj.*` — q/k/v fused into `qkv_proj` at load via
`stacked_params_mapping`, `gpt_oss.py:1100-1106`):

| checkpoint key | dtype | shape | notes |
|---|---|---|---|
| `{p}.weight` | uint8 | `[out, in/2]` | 2 E2M1 nibbles/byte along **input** dim; low nibble = even column |
| `{p}.weight_scale` | float8_e4m3fn | `[out, in/16]` | per-16-group scale, **normalized by weight_scale_2** |
| `{p}.weight_scale_2` | float32 | `[1]` | **ModelOpt convention: `amax(tensor) / (6.0 × 448.0)`** (`modelopt.py:1246-1259`) |
| `{p}.input_scale` | — | — | NOT required on disk for W4A16; placeholder param is deleted after load (`modelopt.py:1345-1361`) |

Dequantization: `W = E2M1(code) × fp8(weight_scale) × weight_scale_2`.

**Fused q/k/v sharp edge:** `process_weights_after_loading` takes
`weight_scale_2.max()` across the fused q/k/v partitions and warns if they
differ (`modelopt.py:1362-1377`) — the fp8 group scales are NOT rescaled, so
**q, k, and v must be quantized with one shared `weight_scale_2`** or
dequantization is wrong for the shards with the smaller global scale.
Kernel: `MarlinNvFp4LinearKernel`, pinned directly (`modelopt.py:1270-1279`)
— no silent W4A4 fallback.

## 4. FusedMoE expert tensors

Created by `ModelOptNvFp4FusedMoE.create_weights` (`modelopt.py:1437-1553`).
vLLM parameter shapes (E = num_experts, H = hidden, I = intermediate;
`w13_num_shards = 2` since `is_act_and_mul`):

| vLLM param | dtype | shape |
|---|---|---|
| `w13_weight` | uint8 | `[E, 2I, H/2]` |
| `w2_weight` | uint8 | `[E, H, I/2]` |
| `w13_weight_scale` | float8_e4m3fn | `[E, 2I, H/16]` |
| `w2_weight_scale` | float8_e4m3fn | `[E, H, I/16]` |
| `w13_weight_scale_2` | float32 | `[E, 2]` (allclose-checked, `[:,0]` used — `modelopt.py:1560-1570`) |
| `w2_weight_scale_2` | float32 | `[E]` |
| `w13_input_scale` | float32 | `[E, 2]` (dropped by Marlin W4A16 path) |
| `w2_input_scale` | float32 | `[E]` (dropped) |
| `w13_bias` / `w2_bias` | bf16 | `[E, 2I]` / `[E, H]` |

**Checkpoint orientation:** `_load_weights_other` narrows then
**`permute(0, 2, 1)`** every key containing `.w13_weight` / `.w2_weight`
(`gpt_oss.py:1020-1047`). Checkpoint tensors are therefore stored in **HF
orientation** (input dim before output dim):

| checkpoint key | dtype | shape | relation to our artifact (`[out,in]` codes) |
|---|---|---|---|
| `...experts.w13_weight` | uint8 | `[E, H/2, 2I]` | per-expert `codes.T` |
| `...experts.w13_weight_scale` | fp8 | `[E, H/16, 2I]` | per-expert `scales.T` |
| `...experts.w2_weight` | uint8 | `[E, I/2, H]` | per-expert `codes.T` |
| `...experts.w2_weight_scale` | fp8 | `[E, I/16, H]` | per-expert `scales.T` |

(The pairs packed along the input dim of our `[out,in]` codes remain packed
along the input dim after transpose — layouts are byte-compatible.)
Columns of w13 stay HF-interleaved (gate even / up odd) per §2.
TP slicing assumes this orientation (`[:, :, 2·tp_start : 2·tp_end]`);
we serve TP=1 so slices are full-size no-ops.

**Backend:** `select_nvfp4_moe_backend(..., activation_key=None)` in W4A16
mode leaves only Marlin (`modelopt.py:1405-1416`;
`fused_moe/oracle/nvfp4.py:118-122, 153, 409`).

## 5. Bias / router / sinks / embeddings

- `...experts.gate_up_proj_bias [E, 2I]` and `...experts.down_proj_bias
  [E, H]` keep HF names (mapper → `w13_bias`/`w2_bias`), loaded by the bias
  branches (`gpt_oss.py:1048-1069`). BF16.
- `...self_attn.sinks [heads]` — mapper → `.attn.sinks`, loaded by the sinks
  branch (`gpt_oss.py:1070-1076`). BF16.
- `model.embed_tokens.weight` → `model.embedding.weight` (mapper). BF16.
- `mlp.router.{weight,bias}` load through the generic path. BF16.

## 6. Sharp edges (verified in source; fixture-tested)

1. **`weight_scale_2` is semantic, not decorative.** Marlin consumes the
   ModelOpt global scale form `amax/2688` "without reciprocation"
   (`modelopt.py:1258-1268, 1379-1388`). fp8 block scales must be stored
   normalized by it. Consequence for the quantizer: per-tensor global scale
   fixed before GPTQ, fp8 block scales = `raw/global` (this also uses the
   full fp8 range instead of drowning small blocks in fp8 subnormals) — and
   **q/k/v must share one global scale** (§3).
2. **Loader bug for 2-D scale_2 MoE keys:** the `_load_weights_other` branch
   fires on the SUBSTRING `".w13_weight" in name` (`gpt_oss.py:1020`), so
   `...w13_weight_scale_2 [E,2]` would enter the 3-D narrow+permute path and
   crash (`weight[:, :, a:b]` on a 2-D tensor). Same for
   `...w2_weight_scale_2 [E]`. `*_input_scale` keys do NOT match the
   substring and load via the generic path. Mitigation: serve-time shim
   (`scripts/vllm_gptoss_nvfp4_shim.py`) that routes `*_weight_scale_2`
   keys to their parameters' weight_loaders ahead of the broken branch.
   Verified empirically by the fixture test; upstream-issue candidate.
3. **Unloaded-parameter check:** vLLM raises if registered checkpoint-backed
   params are missing after load, so `w13_weight_scale_2`, `w2_weight_scale_2`,
   `w13_input_scale`, `w2_input_scale` must be present in the checkpoint
   (input scales as 1.0 placeholders; they are dropped by the W4A16 path).

## 7. Fixture proof

`tests/internalTests/test_vllm_fixture_load.py` builds a tiny GPT-OSS
NVFP4 checkpoint with exactly this contract and loads it in `.venv-serve`
vLLM on the H100, asserting the quant method resolves to `modelopt_fp4`
W4A16, the Marlin kernels engage, and greedy generation runs. Kernel/log
evidence: `logs/serving/fixture_load_*.log`.
