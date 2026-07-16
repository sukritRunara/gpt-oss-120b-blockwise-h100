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

### 🟠 P0.2 — Expert-routing patch is Transformers-version-dependent
**Evidence:** [gpt_oss_expert_gptq.py:176-220](blockwise-gptq-main/opteam-blockwise-gptq/gpt_oss_expert_gptq.py#L176-L220)
derives `num_exp = routing_weights.shape[1]` and indexes `routing_weights[token_idx, e]`
by **expert ID**.
**Analysis:** This is correct **iff** the installed Transformers passes a *dense*
`routing_weights` `[tokens, num_experts]` (scatter of top-k softmax). For the variant
that passes top-k-only weights `[tokens, top_k]`, `num_exp` becomes `top_k` (=4) and
`F.one_hot(router_indices, num_classes=5)` **crashes** for any expert ID ≥ 5 — or worse,
silently mis-weights. Also the patch always runs the "training-path" explicit loop;
numerical equivalence with the pinned version's eval path is unverified.
**Fix:** Pin transformers in `.venv-quant`, read its `GptOssExperts.forward`, match
semantics exactly, add the §12 routing test battery (32 experts / top-4, expert IDs > k,
repeated experts, multi-expert tokens, patched-vs-reference forward equivalence).

### 🔴 P0.3 — `_GptqH` accumulation is mathematically wrong (≠ `GPTQ.add_batch`)
**Evidence:** [expert_dispatch.py:63-77](blockwise-gptq-main/opteam-blockwise-gptq/expert_dispatch.py#L63-L77)
vs [gptq.py:38-55](blockwise-gptq-main/opteam-blockwise-gptq/gptq.py#L38-L55).
**Analysis (verified by hand):** `GPTQ.add_batch` maintains the invariant
`H = (2/N) Σ xxᵀ` — every row weighted equally. `_GptqH.add_batch` adds each batch with
weight `n/N²` (it divides by `√N` *and* multiplies by `n/N`): two batches n₁, n₂ yield
`H = X₁ᵀX₁/N + (n₂/N²)X₂ᵀX₂` — batch-2 rows carry `n₂/N` the weight of batch-1 rows.
Since expert token counts vary per calibration sample, later samples are systematically
underweighted. The docstring's claim that the formula "is identical to GPTQ.add_batch"
is false. All GPT-OSS parallel-mode expert Hessians are affected.
**Fix:** One canonical accumulator used by both paths + chunked==one-shot equivalence
tests with unequal chunk sizes (handoff §12).

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

_(none yet)_
