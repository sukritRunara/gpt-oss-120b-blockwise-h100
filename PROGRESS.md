# PROGRESS

Chronological log of major actions. Newest entries at the top. Never record "done"
without pointing to the test, log, or artifact that proves it.

---

## 2026-07-17 — P0.10 ROOT-CAUSED AND FIXED: full-NVFP4 serving gate PASS

**Status:** fix validated on the full 20B pack; full-NVFP4 D/C serving
benchmarks running (branch `fix/marlin-moe`)

**Root cause:** `ops.moe_wna16_marlin_gemm`'s `mul_topk_weights=True` path
reads a bogus fp32 multiplier at gpt-oss shapes (~1e33 output). Proven by
standalone capture/replay vs an exact torch reference: gemm1 clean; gemm2
`top_k=4, mul=True` → 128/128 rows bad (absmax 3.98e+32); identical args with
`top_k=1, mul=False` + external multiply → 0/128 bad (maxrel 2.4e-3).
Weight-value independent; thread-config/reduce-mode independent; small shapes
unaffected (layout-dependent OOB). Scripts: `scripts/marlin_replay*.py`,
`scripts/fixture_numeric_check.py`.

**Fix (plugin P5, D-014):** gemm2 runs with `mul_topk_weights=False`; routing
weights applied externally (`out.mul_(topk_weights.reshape(-1,1))`).
Mathematically identical, no CUDA rebuild.

**Validation:**
- 2-layer real-weight repro: greedy output now matches its QDQ reference
  (was all-garbage) — `results/pilot/fixture_real2l_*`
- **Full 20B arm-D pack: serving gate PASS** — deterministic, Harmony chat
  coherent, greedy-64 prefix agreement **0.869** (gate ≥0.85) —
  `results/quality/serving_check_fullD.json`

**Docs:** D-014, KNOWN_ISSUES P0.10→resolved,
`docs/UPSTREAM_ISSUE_VLLM_MARLIN_MOE.md` (draft, not filed).

**Completion (same day):** full-NVFP4 C and D benchmarked end-to-end (36
cells, zero failures) — D serves in **12.86 GiB** (vs 39.15 BF16) at ~2× BF16
decode throughput (3397 vs 1717 tok/s @ c=32), slower compute-bound prefill
(expected weight-only trade) — `results/serving/comparison_final.{json,md}`.
Arm C gate: **0.885 PASS**. Gate probe hardened along the way (it had compared
batch-of-8 vs batch-of-1 — batch-size variance, which even BF16 fails; then a
self-matching pgrep deadlock; then prefix-cache default). Four strict probes
characterized an intermittent, quality-neutral near-tie greedy flip in the
Marlin MoE path → KNOWN_ISSUES P1.1, D-015 (reported, not gated).

---

## 2026-07-16 17:00 UTC — NIGHT-1 COMPLETE: all arms built, quality + serving measured

**Status:** complete — see docs/REPORT.md for the full write-up

**Full run (arm D):** 512×2048 C4, 45.6 min, collection peak 49.3 GB
(group=4, D-012), **1632/1632 tensors GPTQ_NVFP4 (100%, zero fallbacks)**,
packed 13 GB, all artifacts bit-exact-verified twice.
**Arm C (matched RTN):** same mask/scales/exporter, 1632/1632, 13 GB packed.

**Quality (authoritative: logit fidelity to B on held-out prompts):**
- D (GPTQ): KL 0.0113, cos_min 0.99743, top-1 1.00 → **2.24× lower KL than C**
- C (RTN): KL 0.0254, cos_min 0.99466, top-1 1.00
- Task suite: B 39/40, C 40/40, D 40/40 — quantization-transparent at task level
- Perplexity: unfit metric for this Harmony model — RTN "improves" raw-text
  ppl while being farther from the source; harness null test Δ=0.0000 proves
  the pipeline exact; documented in REPORT §2 with evidence files.

**Serving (vLLM, TP=1, identical flags, 5+30 req/cell, zero failures):**
- Weights in VRAM: A 13.02 GiB / B 39.15 / D-hybrid 38.22
- A (native MXFP4) leads throughput at high concurrency (3.7k tok/s c=64);
  D-hybrid ≈ B (expected null: experts are BF16 due to P0.10)
- results/serving/comparison_night1.{json,md} + per-request JSONL + telemetry

**Fixes landed during the night:** wikitext dataset namespacing,
transformers-5.x chat-template BatchEncoding, vllm launcher flag,
stage-6 explicit baseline paths, eval-orchestrator resume.

**Open:** P0.10 (upstream vLLM Marlin NVFP4-MoE corruption at E=32×2880) —
fully isolated with kept repro checkpoints; upstream issue candidate.

---

## 2026-07-16 13:45 UTC — Pilot verdict; upstream Marlin MoE bug isolated (P0.10); full run launched

**Status:** full 512×2048 run in progress

**Pilot results (32×512, §13):**
- Stage 5: 26.7 min, 1632/1632 tensors NVFP4 (1610 GPTQ + 22 RTN for
  never-activated experts, all recorded); collection peak **41.4 GB** (P0.4
  proof at 20B scale); zero silent fallbacks.
- Stage 7: all 96 linears + 1536 expert slices packed, 1632/1632 artifacts
  re-verified bit-exact; **13 GB packed vs 39 GB BF16 (3.0×)**.
- QDQ model generates coherent, correct text (greedy references saved).
- vLLM gates: **linears-only hybrid passes** (deterministic, Harmony chat OK,
  greedy agreement 0.90 vs QDQ). **Full-NVFP4 pack fails** — MoE outputs
  ~1e33 → uniform logits.

**P0.10 isolation (3 h of controlled bisection, 9 fixture configurations):**
value-independent, dim-triggered upstream bug in vLLM 0.25.1's Marlin
NVFP4-MoE at exactly (E=32, N=K=2880); every other tested combination —
including (E=8, 2880²) and (E=32, 1024²) — agrees with its QDQ reference,
often exactly. Emulation backend rejects bias; Humming's JIT fails here;
no newer vLLM exists. Full evidence + repro checkpoints in KNOWN_ISSUES
P0.10. Plugin v0.2 landed real fixes along the way (bias kernel-format
conversion, de-interleave + swigluoai_uninterleave, verified on fixtures).

**Decision D-013:** proceed — quality science (B/C/D) on QDQ in transformers;
serving on A (native MXFP4), B (BF16), D-hybrid (labeled); full-NVFP4
serving reported as blocked-by-upstream with repros.

---

## 2026-07-16 10:45 UTC — Arm B built + validated; pilot running

**Status:** in progress (pilot stage 5 collection ~40% at time of writing)

**Arm B (gpt-oss-20b-mxfp4-dequant-bf16):** 39.0 GB, 20.915B params.
- Built by `scripts/dequantize_gpt_oss_20b.py` via the pinned transformers
  `Mxfp4Config(dequantize=True)`. Found + worked around a transformers 5.14
  save bug: `save_pretrained` → `revert_weight_conversion()` silently DROPS
  dequantized expert weights (first attempt produced a 4.9 GB checkpoint
  without experts). Manual sharded save + parameter-count fail-closed check;
  pitfall documented in docs/DEQUANTIZATION_PROVENANCE.md.
- Validation `scripts/validate_dequantized_source.py`: **PASS** —
  structure clean; **all 48 expert tensors decode bit-exact** from A's
  MXFP4 blocks/scales; all 363 other tensors byte-identical; full two-way
  coverage; deterministic forwards; A-vs-B logits identical (A was
  auto-dequantized by transformers → weak check, native-A comparison
  deferred to vLLM). `results/dequant_validation.json`.

**Pilot (§13) in flight** — `scripts/run_pilot.sh`, logs under
`logs/quantization/pilot_*`. Real-model P0.4 evidence already visible:
collection GPU peak **41.4 GB** (vs ~93 GB for the old design → OOM),
~14 s per layer-group pass, 2.18 GB cache per layer.

**Also landed:** serving benchmark harness (P0.9 implementation),
per-arm vLLM launchers/configs, matched-RTN builder (arm C),
logit/task quality eval scripts, full-run script (gated on pilot),
docs (ARCHITECTURE, EXPERIMENT_DESIGN, DEQUANTIZATION_PROVENANCE).

---

## 2026-07-16 10:05 UTC — P0.7 + P0.8 resolved: full-NVFP4 GPT-OSS proven loadable in vLLM

**Status:** complete

**Headline:** vLLM 0.25.1 serves W4A16_NVFP4 GPT-OSS **including the FusedMoE
experts** (Marlin weight-only kernels). A tiny fixture built by the real pipeline
(quantize → manifest → stage 7 pack) loads and generates on the H100:
`FIXTURE_LOAD_OK` (`logs/serving/fixture_load_attempt4.log`).

**What was built:**
- `docs/VLLM_NVFP4_CONTRACT.md` — the full tensor contract (names, shapes, dtypes,
  orientations, scale semantics, loader behavior) with file/line references into the
  installed vLLM source.
- D-010 (quantizer): ModelOpt global-scale convention — per-tensor
  `amax/(6·448)` fp32 global fixed pre-GPTQ, fp8 block scales normalized by it,
  shared across fused q/k/v. Artifacts/manifest/dequant/verify all carry it.
- Stage 7: expert packing to the vLLM FusedMoE layout (per-expert transposed exact
  codes/scales in HF orientation, per-expert scale_2, input-scale placeholders);
  flat `group_size` + vLLM-prefix ignore names in config.json; per-layer fail-closed
  hybrid policy.
- `vllm-gptoss-nvfp4-plugin` (installed editable in .venv-serve): patches three
  upstream gaps — no MoE bias params in ModelOptNvFp4FusedMoE.create_weights,
  no bias/swigluoai constants in its Marlin quant config, and the
  `".w13_weight" in name` loader branch crashing on 2-D scale_2 keys.
- `scripts/build_nvfp4_fixture.py` + `scripts/fixture_load_test.py` — the proof
  harness (attempts 1-3 in logs document each upstream failure in isolation).

**Evidence:** 49 pytest tests green (incl. new global-scale round trip and expert
packing layout assertions); stage/property scripts pass; fixture load log shows
`quantization=modelopt_fp4` + Marlin FP4 linear & MoE kernel engagement.

**Next:** Arm B dequantization script + validation; P0.9 benchmark harness; pilot.

**Blockers:** none.

---

## 2026-07-16 09:30 UTC — P0.5 + P0.6 resolved; official checkpoint downloaded (arm A)

**Status:** complete

**Goal:** (1) Download and pin the official MXFP4 checkpoint. (2) Capture GPTQ's
exact codes/scales and emit the full per-tensor disposition manifest; make Stage 7
serialize exactly those artifacts and fail closed.

**Arm A downloaded:** `openai/gpt-oss-20b` → `/workspace/models/gpt-oss-20b-official-mxfp4`
(38.5 GB incl. metal/ extras), revision pinned + per-file SHA-256 in `PROVENANCE.json`
(`scripts/download_official_model.py`, resumable, HF-504-retrying;
log `logs/setup/download_official_20260716.log`).

**What was built:**
- `quantizer.py` — NVFP4 capture mode (`begin/end/abort_capture`): records exact
  E2M1 nibbles + FP8 scales inside `quantize_dequantize` with column-coverage
  accounting; supports GPTQ blockwise sweeps (incl. partial final blocks) and RTN.
- `quant_artifacts.py` (new) — artifact dataclass, nibble pack/unpack, bit-exact
  `dequantize_artifact` + `verify_artifact_matches`, per-layer safetensors shards
  (atomic), and the P0.5 manifest writer/reader with hard schema validation.
- `apply.py` — Phase 2 captures every linear (verified bit-exact immediately),
  expert handler captures every expert slice (verified against the fp32 shim before
  the bf16 writeback); per-layer artifact streaming; full disposition records.
- `expert_dispatch.py` — GptOss quantize() refactored (`_one_slice`), reasons +
  artifacts per expert; `build_records()` emits manifest records per slice.
- `stage5_quantize_model.py` — emits `quant_artifacts/` + manifest; excluded-params
  list makes records ∪ excluded == all named parameters exactly.
- `stage7_save_modelopt.py` — rewritten: requires the manifest (no fail-open path),
  loads QDQ tensors directly from safetensors (never `from_pretrained` — the old
  code packed init-transformed RAW weights), re-verifies every artifact against the
  checkpoint, packs linears from exact codes/scales, REFUSES expert slices without
  `--allow_hybrid` (then: loud HYBRID label + ignore list + BF16 byte fraction in
  `PACKING_REPORT.json`). Legacy fp8/int8 re-derivation packers removed.

**Evidence:**
- `test_exact_artifacts.py` 11/11 — bit-exact round trips (GPTQ, partial blocks,
  RTN, bf16 cast), non-QDQ-basis repack drift demo, tamper detection.
- `test_manifest_e2e.py` 5/5 — every eligible tensor recorded, artifacts reproduce
  in-model weights bit-exact (incl. expert slices via transpose), coverage exact,
  fail-closed validation, BF16-fallback semantics.
- `test_stage7_exact.py` 5/5 — no-manifest refusal, expert refusal w/o
  --allow_hybrid, exact packed == artifact codes, tampered-artifact abort, dense
  full pack. (`logs/tests/stage7_exact_20260716.log`)
- Full sweep: 48 pytest tests green + stage1 5/5, stage2 7/7, stage3 gpt-oss 6/6,
  stage3 deepseek 56/56, property1/2/4 (`logs/tests/full_pytest_20260716.log`).

**Next:** P0.7/P0.8 — read pinned vLLM 0.25.1 ModelOpt NVFP4 + GPT-OSS loader
source, write `docs/VLLM_NVFP4_CONTRACT.md`, implement expert packing, prove with a
small fixture load in `.venv-serve`. Then the dequantization script (arm B).

**Blockers:** none.

---

## 2026-07-16 08:20 UTC — P0.4: memory-bounded, resumable Hessian collection

**Status:** complete

**Goal:** Replace the OOM-prone all-at-once parallel Hessian collection with the
grouped, cache-backed, resumable design (handoff §P0.4), fail-closed throughout.

**What was built:**
- `opteam-blockwise-gptq/hessian_cache.py` — new: manifest-verified per-layer cache
  (SHA-256 per file, atomic temp+rename writes), immutable hashed calibration-token
  cache, NaN/Inf rejection at save, per-group collection stats (runtime, GPU peak,
  host RSS, bytes).
- `apply.py` — `_run_parallel` rewritten: Phase 1 collects pending layers in groups
  of `--hessian_layer_group_size` (default 1; ~2.4 GB accumulators per group vs
  ~51 GB before), one full-model pass per group over cached tokens; Phase 2
  quantizes from cache with lazy per-layer GPTQ instances. Resume skips
  hash-verified layers and reloads cached tokens. Sample exceptions abort (the old
  code warned and skipped); dense sublayers with zero samples abort; MoE layers
  whose patched forward was never invoked abort (fused-kernel bypass detection).
- `expert_dispatch.py` — `attach_passthrough`/`detach_passthrough`: every collection
  pass pins ALL MoE layers to the collection patch's loop implementation.
- `stage5_quantize_model.py` — `--hessian_cache_dir` (repo-relative default) and
  `--hessian_layer_group_size` CLI args.

**Key finding:** transformers 5.14 dispatches expert forwards via
`use_experts_implementation` ("batched_mm", ULP-different from the eager loop). Before
the passthrough pinning, grouped collection produced Hessians that depended on group
membership — observed as a single E2M1 bin flip in final weights between group sizes.
With pinning, group1 == groupN is **bitwise** (D-008).

**Evidence:**
- `pytest tests/internalTests/test_hessian_grouped_collection.py` → **9/9**
  (`logs/tests/hessian_grouped_20260716.log`): bitwise grouping equivalence (caches
  AND final weights), resume-only-missing-layers (forward-pass count verified),
  layer/token tamper detection, expert-bypass detection, fail-closed aborts,
  stats recording.
- Full regression green: stage1 5/5, stage2 7/7, stage3 gpt-oss 6/6,
  stage3 deepseek 56/56, property1/2/4, routing 9/9, hessian canonical 9/9.

**Next:** P0.5 + P0.6 — per-tensor disposition manifest from Stage 5 and exact
code/scale preservation through Stage 7 (no re-quantization at pack time).

**Blockers:** none.

---

## 2026-07-16 07:35 UTC — P0.2 (expert routing) and P0.3 (Hessian accumulation) fixed

**Status:** complete

**Goal:** Fix the two highest-risk correctness bugs with proof against the pinned
environment.

**P0.2 — expert routing.** Read the installed transformers 5.14.0
`modeling_gpt_oss.py`: router emits top-k-only `router_scores [tokens, top_k]`;
`GptOssExperts.forward` indexes `routing_weights[token_idx, top_k_pos]` (by
POSITION); `one_hot(num_classes=num_experts)`; experts get flat input. The old patch
crashed at one_hot for expert IDs ≥ 5 on this contract. Rewrote
`patch_expert_forward` to match the pinned semantics exactly, with shape-based
dense/top-k contract detection, hard error on unknown contracts, and a
`call_counter` to detect kernel-fusion bypass (GptOssMLP carries
`@use_kernel_forward_from_hub("MegaBlocksMoeMLP")`).

**P0.3 — Hessian accumulation.** Introduced canonical `gptq.accumulate_hessian()`
(invariant `H = (2/N)·Σ xxᵀ` over flattened rows); `GPTQ.add_batch` and
`_GptqH.add_batch` now both delegate to it. Old `_GptqH` weighted chunks `n/N²`.

**Commands / evidence:**
- `pytest tests/internalTests/test_gpt_oss_routing.py` → **9/9**
  (`logs/tests/gpt_oss_routing_20260716.log`) — incl. patched≡reference equivalence
  against the real transformers class
- `pytest tests/internalTests/test_hessian_canonical.py` → **9/9**
  (`logs/tests/hessian_canonical_20260716.log`) — incl. unequal-chunk equivalence,
  `_GptqH ≡ GPTQ`, transplant quantization identity
- Regression: stage1 5/5, stage2 7/7, stage3 6/6, property1/2/4 all pass

**Files changed:**
- `opteam-blockwise-gptq/gpt_oss_expert_gptq.py` — patch rewritten (P0.2)
- `opteam-blockwise-gptq/gptq.py` — `accumulate_hessian()` added; `add_batch` delegates
- `opteam-blockwise-gptq/expert_dispatch.py` — `_GptqH.add_batch` delegates
- `tests/internalTests/test_gpt_oss_routing.py` — new (9 tests)
- `tests/internalTests/test_hessian_canonical.py` — new (9 tests)
- `DECISIONS.md` — D-005 (routing contract), D-006 (Hessian convention)

**Next:** P0.4 — memory-bounded, resumable layer-group Hessian collection
(`--hessian_layer_group_size`), plus fail-loud calibration coverage checks.

**Blockers:** none.

---

## 2026-07-16 07:12 UTC — Environments frozen; P0.1 path fix landed and verified

**Status:** complete

**Goal:** Finish §22 items 5–7: both environments bootstrapped with lockfiles, system
manifest frozen, hard-coded paths repaired and proven portable.

**Commands:**
- `scripts/bootstrap_quant_env.sh` → `.venv-quant` (log: `logs/setup/bootstrap_quant.log`)
- `scripts/bootstrap_serve_env.sh` → `.venv-serve` (log: `logs/setup/bootstrap_serve.log`)
- `scripts/capture_system_manifest.sh` → `envs/system-manifest.txt`
- Stage 1–3 CPU tests run **from outside the repo root** (P0.1 portability proof):
  - `stage1_nvfp4_unit_tests.py` → 5/5 PASS (`logs/tests/stage1_20260716.log`)
  - `stage2_nvfp4_algorithm_tests.py` → 7/7 PASS (`logs/tests/stage2_20260716.log`)
  - `stage3_gpt_oss_shape_tests.py` → 6/6 PASS (`logs/tests/stage3_gpt_oss_20260716.log`)
- `grep -rn "runara_dgx|/home/"` → zero hits

**Environment (frozen in envs/*.lock.txt + system-manifest.txt):**
- `.venv-quant`: torch 2.13.0+cu130, transformers 5.14.0, datasets 5.0.0,
  safetensors 0.8.0, accelerate 1.14.0, kernels 0.16.0, triton 3.7.1
- `.venv-serve`: vllm 0.25.1, torch 2.11.0+cu130, transformers 5.14.0,
  flashinfer 0.6.13 — note the torch version differs from .venv-quant,
  which is exactly why two isolated environments are mandatory.

**Files changed:**
- 10 × `tests/**.py` — `_CODE_ROOT` now `Path(__file__).resolve().parents[N] / "opteam-blockwise-gptq"`
- `tests/internalTests/test_vLLM_deploy_quantized_model.py` — hard-coded MODEL path →
  required `--model` CLI arg
- `envs/{quant,serve}-requirements.lock.txt`, `envs/system-manifest.txt` — new

**Next:** P0.2 — inspect installed transformers 5.14.0 `GptOssExperts.forward`, pin the
routing contract, write the routing test battery, fix the expert patch. Then P0.3
(canonical Hessian accumulator + equivalence tests).

**Blockers:** none.

---

## 2026-07-16 07:05 UTC — §22 first actions: branch, recon, static P0 audit, env bootstrap

**Status:** complete (see follow-up entry above for env results)

**Goal:** Execute handoff §22 items 1–7: recon snapshot, working branch, static audit
of all 9 documented P0 issues against the real source, environment scaffolding.

**Commands:**
- Recon (`pwd`, tree, `git`, `nvidia-smi`, `df`, `free`) → `logs/setup/recon_20260716.log`
- `git switch -c h100-gpt-oss-20b-nvfp4` + pushed to origin
- Read/grepped all core sources: `apply.py`, `gptq.py`, `quantizer.py`,
  `expert_dispatch.py`, `gpt_oss_expert_gptq.py`, `model_utils.py`,
  `stage5_quantize_model.py`, `stage7_save_modelopt.py`, `stage8_benchmark_*.py`
- `scripts/bootstrap_quant_env.sh` / `bootstrap_serve_env.sh` (running,
  logs in `logs/setup/bootstrap_{quant,serve}.log`)

**Results — audit (all 9 P0s verified with file/line evidence in KNOWN_ISSUES.md):**
- P0.1 confirmed: hard-coded DGX `_CODE_ROOT` + RuntimeError guard in 10 files —
  nothing under `tests/` imports on this pod.
- P0.2 version-dependent: expert patch assumes *dense* `routing_weights
  [tokens, num_experts]`; crashes/mis-weights for the top-k-only Transformers variant.
- P0.3 confirmed mathematically: `_GptqH` weights batches `n/N²` vs `GPTQ.add_batch`
  invariant `2/N` — later calibration batches systematically underweighted.
- P0.4 confirmed: ~51 GB expert Hessians allocated **on GPU** during single-pass
  collection + ~42 GB model > 80 GB H100. Cache is written only after the pass.
- P0.5 confirmed: stage5 JSON has no `quantized_attn_keys`/`layer_losses`;
  stage7 requires them.
- P0.6 confirmed: `pack_nvfp4` re-derives scales from QDQ weights (independent second
  quantization → packed ≠ GPTQ-optimized).
- P0.7 confirmed: packing iterates `nn.Linear` only → GPT-OSS batched experts can never
  pack; missing manifest triggers fail-open "pack everything".
- P0.8 confirmed: `weight_scale_2=1.0` Marlin assumption unverified; shared-scale
  computation dead code.
- P0.9 confirmed: TTFT = offline `generate(max_tokens=1)`; no server/concurrency.

**Files changed:**
- `KNOWN_ISSUES.md` — all 9 P0s moved from "unverified" to confirmed with evidence
- `envs/{quant,serve}-requirements.in`, `scripts/bootstrap_{quant,serve}_env.sh`,
  `scripts/capture_system_manifest.sh` — new

**Next:** Freeze lockfiles + system manifest when installs finish; then P0.1 path fix
(smallest, unblocks all tests) → P0.2/P0.3 with tests.

**Blockers:** none.

---

## 2026-07-16 06:42 UTC — Repo bootstrap and project scaffolding

**Status:** complete

**Goal:** Create the GitHub repo, import the `blockwise-gptq` source, and stand up
the progress-tracking / documentation scaffolding before any engineering work.

**Commands:**
- `git add -A && git commit` — initial import (commit `b8329ba`)
- `git push origin main` — pushed to `github.com/sukritRunara/gpt-oss-120b-blockwise-h100`
- `nvidia-smi`, `python3 --version`, `df -h /workspace`, `free -h` — environment probe

**Files changed:**
- `.gitignore` (root) — excludes weights, venvs, caches, logs, secrets, local Claude settings
- `PROGRESS.md`, `DECISIONS.md`, `KNOWN_ISSUES.md` — new
- `README.md` (root) — rewritten to describe the H100 project accurately

**Results / environment (RunPod H100):**
- GPU: 1× NVIDIA H100 80GB HBM3, driver 580.126.09, CUDA 13.0
- Host: Python 3.12.3, ~2 TB RAM, `/workspace` persistent (199 TB free)
- Source tree present under `blockwise-gptq-main/` (code, tests, results, scripts)
- Remote push verified: `main` is on GitHub.

**Next:** Perform handoff §22 immediate actions — audit the source tree, create the
`h100-gpt-oss-20b-nvfp4` working branch, bootstrap the two Python environments
(`.venv-quant`, `.venv-serve`), and freeze a system manifest.

**Blockers:** none.
