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

### 🔴 P0.4 — Parallel-mode expert Hessians OOM the H100
**Evidence:** [apply.py:494-531](blockwise-gptq-main/opteam-blockwise-gptq/apply.py#L494-L531)
allocates accumulators for **all** layers before one full-model pass; `_GptqH` lazily
allocates `H` on `inp.device` — i.e. **on the GPU** during forward.
**Math:** GPT-OSS-20B: 24 layers × 32 experts × 2 proj × (2880² × 4 B) ≈ **51 GB** of
Hessians on GPU + ~42 GB BF16 model ≈ 93 GB > 80 GB H100. `_save_hessians(free_after_save=True)`
only runs **after** the pass completes — it caps RAM, not peak collection VRAM.
**Fix:** Memory-bounded layer-group collection (`--hessian_layer_group_size`, start 1),
resumable per-layer cache with completeness manifest (handoff §P0.4 design).

### 🔴 P0.5 — Stage 5 results JSON lacks the fields Stage 7 requires
**Evidence:** Stage 5 writes only settings + `total_gptq_loss` + timing
([stage5_quantize_model.py:425-442](blockwise-gptq-main/tests/stage5_quantize_model.py#L425-L442)).
Stage 7 looks for `results["quantized_attn_keys"]` and `results["layer_losses"]`
([stage7_save_modelopt.py:683](blockwise-gptq-main/tests/stage7_save_modelopt.py#L683)) —
never produced by Stage 5 → falls through to the fail-open path (see P0.7).
**Fix:** Full per-tensor disposition manifest (schema per handoff §P0.5); Stage 7 must
hard-error on missing fields.

### 🔴 P0.6 — Stage 7 re-quantizes instead of preserving GPTQ's exact codes
**Evidence:** `pack_nvfp4` recomputes `amax/6.0 → FP8 scale → E2M1 snap` from the QDQ
BF16 weights ([stage7_save_modelopt.py:117-179](blockwise-gptq-main/tests/stage7_save_modelopt.py#L117-L179)).
**Analysis:** Stage 5's QDQ values are `grid × scale_gptq`, but a block's amax only
recovers `scale_gptq` if some value hit grid-max 6.0 in that block; otherwise the FP8
re-cast yields a different scale → different codes → the packed model is **not** the
model GPTQ optimized (and not the model Stage 6 evaluated).
**Fix:** Stage 5 must emit exact codes/scales (`QuantizedTensorArtifact`); Stage 7
serializes them; invariant test `QDQ ≈ dequant(packed)`.

### 🔴 P0.7 — Stage 7 cannot pack GPT-OSS experts, and fails open
**Evidence:** Packing iterates `model.named_modules()` with
`isinstance(module, nn.Linear)` ([stage7_save_modelopt.py:478-480](blockwise-gptq-main/tests/stage7_save_modelopt.py#L478-L480)).
GPT-OSS experts are batched `nn.Parameter`s (`experts.gate_up_proj [E,in,out]`) —
invisible to both this loop and `find_layers()`; they'd be stored BF16 while
`config.json` claims `W4A16_NVFP4`. Additionally, with no/insufficient results JSON,
Stage 7 "packs all nn.Linear except lm_head" — the exact fail-open the handoff forbids.
Expert-quantized detection also reads `layer_losses.get("experts.gate_up") != "BF16"`,
a per-layer blanket that ignores per-expert RTN/BF16/quantized mixtures.
**Fix:** Expert-aware packing against the pinned vLLM contract; hard-error without a
complete manifest; per-slice dispositions.

### 🔴 P0.8 — NVFP4 packing contract unverified against pinned vLLM
**Evidence:** `weight_scale_2 = 1.0` hard-coded with an in-comment claim about Marlin
W4A16 behavior ([stage7_save_modelopt.py:156-165](blockwise-gptq-main/tests/stage7_save_modelopt.py#L156-L165));
`_compute_shared_scales` computes group scales then never uses them for packing.
**Fix:** Inspect the pinned vLLM's ModelOpt/NVFP4 loader + kernel selection; write
`docs/VLLM_NVFP4_CONTRACT.md` with file/line references; round-trip + minimal-load tests;
capture server logs proving the selected kernel.

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
