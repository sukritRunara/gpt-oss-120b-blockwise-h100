# KNOWN ISSUES

Open issues from the agent handoff (§10) plus anything found during work.
Each item: severity (P0/P1/P2), repro/evidence, hypothesis, next action, and whether
it invalidates current artifacts. Close items by moving them to "Resolved" with proof.

Status key: 🔴 open (confirmed) · 🟠 open (version-dependent) · 🟡 in progress · 🟢 resolved

All 9 P0s below were **verified against the actual source on 2026-07-16** (static
audit; see PROGRESS.md entry). None are fixed yet. Any artifact produced by the
current pipeline is invalid for the experiment.

---

## Open

### 🔴 P0.1 — Hard-coded DGX paths break every entry point
**Evidence:** `_CODE_ROOT = Path("/home/runara_dgx_spark_1/Itamar/projects/...")` with a
`RuntimeError` guard in 10 files (all `tests/stage*.py`, all `tests/internalTests/*.py`,
e.g. [stage5_quantize_model.py:59-66](blockwise-gptq-main/tests/stage5_quantize_model.py#L59-L66)).
Hard-coded model paths in `test_vLLM_deploy_quantized_model.py:2,35`.
**Impact:** Nothing under `tests/` even imports on this pod.
**Fix:** `_CODE_ROOT = _REPO_ROOT / "opteam-blockwise-gptq"`; portability smoke test.
Note stage7's `_CODE_ROOT` points at the repo root (not `opteam-blockwise-gptq/`) —
the two path constants are inconsistent with each other as well.

### 🔴 P0.9 — Benchmark is single-process, not a serving benchmark
**Evidence:** TTFT = offline `llm.generate(max_tokens=1)` wall-clock averaged over a few
repeats ([stage8_benchmark_nvfp4_vs_bf16.py:117-123](blockwise-gptq-main/tests/stage8_benchmark_nvfp4_vs_bf16.py#L117-L123));
no server, no concurrency, no async, no percentiles, no per-request records.
**Fix:** Async benchmark against a live OpenAI-compatible vLLM server per handoff §18
(JSONL per request, warmup, 3 repetitions, p50/p90/p99, concurrency sweeps).

---

## Notes (non-P0 observations from the audit)

- `apply.py` `_run_parallel` phase-1 hook path wraps sample forwards in
  `try/except Exception` with only a warning — a systematically failing sample would
  silently produce a Hessian from fewer samples than reported. Tighten during P0.4 work.
- `stage7` copies `*.py` custom code and support files from the Stage 5 model dir; fine
  for DeepSeek, but GPT-OSS provenance/manifest requirements (handoff §19) need more.
- `model_utils.get_model_layers` detects GPT-OSS purely by `mlp.experts` presence —
  acceptable, but verify against the pinned Transformers class names during P0.2 work.
- Root `README.md` inside `blockwise-gptq-main/` still documents the unsafe quick-start
  and DeepSeek-era details; superseded by the repo-root README until the pipeline is
  repaired.

---

## Resolved

### 🟢 P0.7 + P0.8 — Expert packing & vLLM contract (resolved 2026-07-16)
**Verdict:** vLLM 0.25.1 supports W4A16_NVFP4 for BOTH GPT-OSS dense linears (FP4
Marlin) and FusedMoE experts (Marlin MoE) — full-NVFP4 GPT-OSS is servable on H100.
The complete tensor contract (names, shapes, dtypes, orientations, scale conventions)
is documented with file/line references in `docs/VLLM_NVFP4_CONTRACT.md`, verified
against the installed source.
**What was fixed on our side:**
- Stage 7 packs expert slices per layer into the FusedMoE checkpoint layout (HF
  orientation, per-expert transposed exact codes/scales, per-expert `weight_scale_2`,
  1.0 input-scale placeholders). Fail-closed: partially-quantized expert layers
  refuse to pack without `--allow_hybrid`.
- D-010: quantizer adopted the ModelOpt scale convention — per-tensor
  `global_scale = amax/(6·448)` with fp8 block scales normalized by it (also fixes
  fp8-subnormal precision loss), **shared across fused q/k/v** (vLLM applies
  `max(weight_scale_2)` to the fused weight without rescaling groups).
- config.json: `quant_algo`/`group_size`/`ignore` written flat (where vLLM reads
  them); ignore names translated to vLLM prefixes.
**Upstream gaps found (patched via `vllm-gptoss-nvfp4-plugin`, loaded through the
`vllm.general_plugins` entry point in every vLLM process):**
1. `ModelOptNvFp4FusedMoE.create_weights` registers no bias params although GPT-OSS
   MoE has biases → KeyError at load.
2. Its Marlin quant config carries neither the biases nor the swigluoai constants
   (α=1.702, β=1.0, clamp=7.0) — experts would silently run plain SiLU without bias.
3. `GptOssModel._load_weights_other` branches on the substring `".w13_weight"`, so
   2-D `*_weight_scale_2` tensors would crash the 3-D permute path.
**Proof:** tiny end-to-end fixture (real pipeline: quantize → manifest → stage7 pack
→ vLLM load on H100): `quantization=modelopt_fp4`, Marlin FP4 kernels engaged for
linear + MoE, generation ran — `FIXTURE_LOAD_OK`
(`logs/serving/fixture_load_attempt4.log`; failures 1-3 documented in attempts 1-3).
Numerical validation of served outputs happens at the pilot logit-comparison gate.

### 🟢 P0.5 — Stage 5 tensor manifest (resolved 2026-07-16)
Was: Stage 5 JSON had only settings + total loss; Stage 7 expected fields that never
existed and fell open to "pack all nn.Linear".
**Fix:** Stage 5 (nvfp4 + parallel mode) now emits `quant_artifacts/manifest.json`
with one record per tensor/expert-slice — name, kind, layer, projection, expert
index, shape, orientation, dtype, disposition (GPTQ_NVFP4 / RTN_NVFP4 /
BF16_FALLBACK), reason, block sizes, loss + normalized loss, Hessian sample count,
artifact reference — plus an excluded-parameters list so records ∪ excluded ==
all named parameters exactly. `read_quant_manifest` hard-errors on any missing
field. **Proof:** `test_manifest_e2e.py` 5/5 (record completeness, coverage,
round-trip, fail-closed validation, BF16-fallback semantics).

### 🟢 P0.6 — Exact codes/scales preserved through export (resolved 2026-07-16)
Was: Stage 7 re-derived NVFP4 scales/codes from weights — and worse, from the RAW
on-disk (potentially init-transformed) weights, not the QDQ tensors GPTQ produced.
**Fix:** `NVFP4Quantizer` capture mode records the exact E2M1 codes + FP8 scales as
`fasterquant_blockwise`/RTN produce them (`begin/end/abort_capture`);
`quant_artifacts.py` stores them per layer (safetensors shards, atomic) and enforces
the invariant `dequantize(codes, scales) == QDQ weight` **bit-exact** — verified
immediately after each tensor quantizes AND again at Stage 7 pack time against the
on-disk checkpoint. Stage 7 never re-quantizes; it serializes the stored artifacts
(legacy fp8/int8 re-derivation packers removed). **Proof:** `test_exact_artifacts.py`
11/11 (round trips incl. partial GPTQ blocks, RTN, bf16 cast; tamper detection;
non-QDQ-basis repack drift demo), `test_stage7_exact.py` 5/5 (exact consumption,
fail-closed, tampered-artifact abort, dense full pack).

### 🟢 P0.1 — Hard-coded DGX paths (resolved 2026-07-16)
Was: `_CODE_ROOT = Path("/home/runara_dgx_spark_1/...")` + RuntimeError guard in 10
files; nothing under `tests/` imported on this pod.
**Fix:** repo-relative `Path(__file__).resolve().parents[N] / "opteam-blockwise-gptq"`;
`test_vLLM_deploy_quantized_model.py` takes a required `--model` arg.
**Proof:** stage1 5/5, stage2 7/7, stage3 6/6 run in `.venv-quant` from a cwd *outside*
the repo root (`logs/tests/stage{1,2,3}_*.log`); `grep 'runara_dgx|/home/'` clean.
Commit `c801d6a`.

### 🟢 P0.2 — GPT-OSS expert-routing bug (resolved 2026-07-16)
Was: patch derived `num_experts` from `routing_weights.shape[1]` and indexed weights by
expert ID. Against the **pinned transformers 5.14.0** (verified in
`.venv-quant/.../models/gpt_oss/modeling_gpt_oss.py`): the router returns top-k-only
`router_scores [tokens, top_k]`, experts index `routing_weights[token_idx, top_k_pos]`,
one_hot uses `num_classes=num_experts`, and experts receive already-flat input. The old
patch therefore **crashed at one_hot** (`num_classes=top_k+1=5`) for any expert ID ≥ 5.
**Fix:** `patch_expert_forward` reimplements the pinned forward exactly (positional
weight lookup, `num_classes=num_experts`, flat-shape preservation), auto-detects the
dense-vs-top-k contract by shape and hard-errors on unknown contracts, and exposes a
`call_counter` so callers can fail loudly if a fused kernel forward
(`@use_kernel_forward_from_hub("MegaBlocksMoeMLP")`) bypasses the patch.
**Proof:** new `tests/internalTests/test_gpt_oss_routing.py` — 9/9 pass incl.
patched-vs-reference numerical equivalence against the real transformers class,
expert IDs ≥ top_k, positional-weight isolation, exact add_batch token subsets
(`logs/tests/gpt_oss_routing_20260716.log`).

### 🟢 P0.4 — Parallel-mode Hessian collection OOM (resolved 2026-07-16)
Was: all-layer accumulators (~51 GB of expert Hessians **on GPU** for GPT-OSS-20B)
allocated for one full-model pass, + ~42 GB model > 80 GB H100; cache written only
after the pass.
**Fix:** grouped collection (`--hessian_layer_group_size`, default 1): accumulators
attach to ≤ group_size layers per full-model pass; each group streams to a
manifest-verified cache (`hessian_cache.py`: atomic writes, SHA-256 per layer file,
immutable hashed calibration-token cache) and frees before the next group. Peak
accumulator memory = one group (~2.4 GB for GPT-OSS-20B at group=1) instead of ~51 GB.
Resume: completed layers (hash-verified) are skipped; tokens reload from the cache.
Fail-closed: sample exceptions abort; zero-sample dense sublayers abort; NaN/Inf H
rejected at save; expert-patch bypass (fused-kernel dispatch) detected via call
counter and aborts. Collection stats (runtime, GPU peak, host RSS, cache bytes)
recorded per group in the manifest.
**Key finding during work:** transformers 5.14 dispatches expert forwards via
`use_experts_implementation` ("batched_mm" etc., ULP-different from the eager loop),
so collection now pins ALL MoE layers to the patch's loop implementation every pass
(`attach_passthrough`) — otherwise Hessians depend on group membership.
**Proof:** `tests/internalTests/test_hessian_grouped_collection.py` — 9/9 incl.
**bitwise** group1==groupN equivalence of caches AND final quantized weights, resume
recollects only missing layers with cached tokens, tamper detection (layer + token
cache), bypass detection, fail-closed aborts
(`logs/tests/hessian_grouped_20260716.log`). Full regression green: stage1 5/5,
stage2 7/7, stage3 gpt-oss 6/6, stage3 deepseek 56/56, property1/2/4, routing 9/9,
hessian canonical 9/9.

### 🟢 P0.3 — `_GptqH` accumulation mathematically wrong (resolved 2026-07-16)
Was: `_GptqH.add_batch` weighted each chunk `n/N²` instead of the uniform per-row
`2/N` — with two batches n₁,n₂: `H = X₁ᵀX₁/N + (n₂/N²)X₂ᵀX₂`, so later expert batches
were systematically underweighted (expert token counts always vary per sample).
**Fix:** single canonical `gptq.accumulate_hessian()` maintaining `H = (2/N)·Σ xxᵀ`
over flattened rows; both `GPTQ.add_batch` and `_GptqH.add_batch` delegate to it. The
row-count convention is documented in its docstring.
**Proof:** new `tests/internalTests/test_hessian_canonical.py` — 9/9 pass incl.
unequal-chunk == one-shot, `_GptqH ≡ GPTQ` on the same stream, direct-formula check,
transplant-into-GPTQ quantization identity, save/load round trip
(`logs/tests/hessian_canonical_20260716.log`). Existing property2 suite still passes.
