# Troubleshooting

Issues actually hit during this project, with the fix that worked.

## Setup / environment

**HF Hub 504 Gateway Timeout during downloads.** Transient hub outages.
`scripts/download_official_model.py` retries with exponential backoff
(6 attempts); it is idempotent — just rerun it.

**`ninja` not found when vLLM starts (engine-core subprocess).** The pip
`ninja` lives in `.venv-serve/bin`, which is not on PATH when the venv's
python binary is invoked directly. Prefix `PATH="/workspace/.venv-serve/bin:$PATH"`
(scripts/serve_vllm.sh does this).

**pytest collects legacy script-style tests and errors.** The property1/2/4
scripts take positional args from their own main(); run them as scripts, or
scope pytest to the pytest-native batteries.

## Quantization

**Everything under tests/ raised `RuntimeError: Code root not found`.**
Hard-coded DGX paths (P0.1) — fixed; paths now resolve relative to the repo.

**Expert calibration crashed at `F.one_hot` (or produced empty Hessians).**
The routing patch didn't match the installed transformers' contract (P0.2)
and MoE forwards can be dispatched to fused implementations that bypass the
patch. Fixed in `patch_expert_forward`; the collection loop hard-errors if
the patch was never invoked. If a future transformers changes the routing
contract, `test_reference_contract_is_topk` fails first — update the patch.

**OOM during parallel Hessian collection.** The legacy all-layer design
needed ~93 GB. Use grouped collection (default `--hessian_layer_group_size 1`
≈ 41 GB peak on the 20B). Raise group size only with measured headroom.

**Hessian results depend on which layers were collected together.**
transformers dispatches expert forwards (`batched_mm` etc.) that are
ULP-different from the eager loop. Collection pins ALL MoE layers to one
implementation per pass (D-008); groupings are bitwise-equivalent.

**Quantized model differs from what Stage 6 evaluated.** Never re-derive
codes/scales at pack time; Stage 7 serializes the captured exact artifacts
and verifies them bit-exact against the QDQ checkpoint (P0.6). If the
verification trips, the artifacts and checkpoint are out of sync — rerun
Stage 5; do not "fix" by re-quantizing.

## Dequantization

**Dequantized save was 4.9 GB with no expert weights.** transformers 5.14
`save_pretrained` → `revert_weight_conversion()` drops dequantized expert
tensors. `scripts/dequantize_gpt_oss_20b.py` saves the state dict manually
and fails closed on parameter-count mismatch.

## Serving (vLLM 0.25.1 + GPT-OSS NVFP4)

**`KeyError: 'layers.N.mlp.experts.w2_bias'` at load.** Upstream:
`ModelOptNvFp4FusedMoE.create_weights` registers no bias params. Fixed by
the `vllm-gptoss-nvfp4` plugin (must be pip-installed in `.venv-serve`;
verify with `pip show vllm-gptoss-nvfp4`).

**Experts numerically wrong (plain SiLU, no bias).** Upstream: the Marlin
MoE quant config lacks biases and swigluoai constants. Same plugin fixes it.

**Loader crash on `*_weight_scale_2` (too many indices).** Upstream:
`_load_weights_other` substring-matches `.w13_weight` and 3-D-permutes 2-D
tensors. Same plugin routes those keys directly.

**Checkpoint won't quantize-load (`quant_algo` missing).** vLLM reads
`quant_algo`, `group_size`, `ignore` FLAT from `quantization_config`
(not from `config_groups`). Stage 7 writes the correct layout.

**Fused q/k/v accuracy warning (`weight_scale_2 differs`).** The quantizer
must share one global scale across q/k/v (D-010). Rebuild the artifact with
the current stage 5 — old artifacts with per-tensor q/k/v globals will
dequantize wrong under vLLM's max() fusion.
