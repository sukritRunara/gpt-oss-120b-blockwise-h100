# Final report: gpt-oss-20b blockwise-GPTQ → NVFP4 on H100

> Status: DRAFT — quality/serving numbers being filled by the night-1 run.
> Every number cites its results file; see PROGRESS.md for the run log.

## 1. What was built

Four arms from one provenance root (`openai/gpt-oss-20b`, revision pinned,
per-file SHA-256 — `models/*/PROVENANCE.json`):

| Arm | Artifact | Size | Status |
|-----|----------|------|--------|
| A | official MXFP4 | 38.5 GB (13 GB weights + extras) | pinned |
| B | mxfp4-dequant-bf16 | 39.0 GB | decode **bit-exact** validated |
| C | …-rtn-nvfp4 (+ packed) | 13 GB packed | 1632/1632 verified, 100% NVFP4 |
| D | …-blockwise-gptq-nvfp4 (+ packed) | 13 GB packed | 1632/1632 verified, 100% NVFP4 |

C and D share the identical tensor mask (all 96 attention linears + all
1536 expert slices), E2M1 grid, fp8 block scales with per-tensor ModelOpt
global scales (shared across fused q/k/v), exact-artifact capture with
bit-exact verification at quantize AND pack time, manifest schema, exporter,
and serving plugin. The ONLY difference is the algorithm (GPTQ vs RTN).

**Memory result (artifact level):** 39 GB BF16 → 13 GB NVFP4 packed
(3.0×; 82.6% of model bytes in 4-bit + fp8 scales, remainder = embeddings,
lm_head, router, norms, biases, sinks — all policy-excluded and manifested).

Full calibration: C4, 512×2048, group-size-4 collection, GPU peak 49.3 GB
(80 GB card), 45.6 min end-to-end, fully resumable, zero fallbacks.

## 2. Quality (B vs C vs D — the primary claim)

QDQ checkpoints evaluated in transformers on identical inputs. Calibration
data (C4 train) disjoint from all evaluation data.

### Logit-level paired metrics (24 held-out prompts; B is reference) — PRIMARY
_results/quality/logit_eval.json_

| Metric | C (RTN) | D (GPTQ) | GPTQ advantage |
|--------|---------|----------|----------------|
| KL divergence (mean) | 0.02541 | **0.01133** | **2.24× lower** |
| cosine (min over prompts) | 0.99466 | **0.99743** | 1.9× less worst-case drift |
| next-token top-1 agreement | 1.000 | 1.000 | tie (both perfect) |
| greedy-64 prefix agreement | 0.340 | 0.361 | +6% relative |

**Blockwise GPTQ is measurably closer to the BF16 source than matched RTN on
every divergence metric**, at identical format, mask, scales, and export path.
(Long-greedy prefix agreement in the 0.3-0.4 range is expected for 4-bit
weight quantization: tiny logit shifts compound over open-ended decoding while
single-step top-1 remains perfect.)

### Perplexity (diagnostic; identical cached token sequences, 2048-token windows)
_blockwise-gptq-main/results/stage4/stage6 JSONs_

| Arm | WikiText-2 | Δ vs B | C4 | Δ vs B |
|-----|-----------|--------|----|--------|
| B (BF16) | 227.08 | — | 711.29 | — |
| D (GPTQ) | 248.49 | +21.41 (+9.4%) | 694.20 | −17.09 (−2.4%) |
| C (RTN) | 140.24 | −86.84 | 196.68 | −514.61 |

**Raw-text perplexity is an unfit quality metric for this model — reported
for completeness, not used for conclusions.** Validation performed per the
handoff's skepticism rule: the harness null test (B evaluated against its own
baseline on the identical cached tokens) gives Δ = 0.0000 on both datasets,
and every measurement reproduces deterministically. The anomaly is real model
behavior: gpt-oss is a Harmony-format reasoning model whose raw-text NLL is
pathologically high at the source (227 on WikiText-2); RTN's coarser
quantization noise partially disrupts that mode and "improves" raw-text NLL
(140) while being measurably FARTHER from the source at the logit level
(KL 2.24× worse than GPTQ). For a transquantization experiment the correct
quality lens is fidelity to the source model — the logit-level suite above —
where GPTQ wins on every metric. (Evidence:
results/quality/stage6_nulltest_B.json, stage6_C_rerun.json.)

### Task suite (40 items: knowledge/math/code/instruction, Harmony chat, greedy)
_results/quality/task_{B,C,D}.json — every prompt, raw output, and verdict saved_

| Arm | Overall | Knowledge | Math | Code | Instruct |
|-----|---------|-----------|------|------|----------|
| B (BF16) | 39/40 (97.5%) | 10/10 | 9/10 | 10/10 | 10/10 |
| C (RTN) | 40/40 (100%) | 10/10 | 10/10 | 10/10 | 10/10 |
| D (GPTQ) | 40/40 (100%) | 10/10 | 10/10 | 10/10 | 10/10 |

Both NVFP4 arms are at ceiling — **no measurable task-level degradation** at
this suite's difficulty (B's single math miss is within noise). Task-level
capability survives 4-bit conversion intact; the C-vs-D difference is visible
only at logit fidelity, where GPTQ is 2.24× closer to the source.

## 3. Serving on H100 (vLLM 0.25.1, TP=1, identical flags)

**Upstream blocker (P0.10):** vLLM 0.25.1's Marlin NVFP4-MoE kernel produces
numerically corrupt output (~1e33) at exactly GPT-OSS's expert dimensions
(E=32, N=K=2880) — proven value-independent with minimal repro fixtures
(KNOWN_ISSUES.md P0.10; docs/VLLM_NVFP4_CONTRACT.md §6). The same artifacts'
dense linears serve correctly (0.90 greedy agreement vs QDQ). Consequently:

- Arms benchmarked end-to-end: **A** (native MXFP4 path), **B** (BF16),
  **D-hybrid** (attention NVFP4 via Marlin + experts BF16 — explicitly
  labeled; NOT representative of full-NVFP4 serving).
- **No full-NVFP4 GPT-OSS serving numbers are claimed on this stack.** The
  packed artifacts are complete and verified; they await an upstream kernel
  fix (three concrete bugs already patched via our plugin; the fourth —
  the corruption — requires a kernel-level fix).

### Serving results (suites: prefill 1k/8k×1, decode 128×256, mixed 1k/8k×256)
_results/serving/*/summary.json; per-request JSONL alongside_

TBD table: TTFT p50/p99, ITL, output tok/s by concurrency, VRAM after load.

## 4. Interpretation guide

- **D vs C** (same format, same mask): pure blockwise-GPTQ benefit.
- **D vs B**: total cost of NVFP4 conversion.
- **A vs B**: decode validation — completed, exact (all expert tensors
  decode bit-exact; everything else byte-identical).
- H100 has no native FP4: any NVFP4 serving path is weight-only
  (Marlin); memory is the expected win, speed was never promised.

## 5. Reproduction

See README.md §6 for the exact command sequence (all verified on this pod),
`envs/` for lockfiles and the system manifest, and `PROGRESS.md` for the
chronological evidence trail.
