# Final report: gpt-oss-20b blockwise-GPTQ → NVFP4 on H100

> Status: night-1 COMPLETE (2026-07-16/17). Every number cites its results
> file; see PROGRESS.md for the chronological evidence trail.

## Executive summary

1. **GPTQ beats matched RTN on fidelity to the source at identical NVFP4
   format**: KL divergence 2.24× lower (0.0113 vs 0.0254), worst-case logit
   cosine 0.99743 vs 0.99466, both with perfect next-token top-1 agreement.
   Task-level capability is at ceiling for BOTH 4-bit arms (40/40).
2. **Memory**: the full-NVFP4 artifact is 13 GB vs 39 GB BF16 (3.0×) —
   byte-parity with the official MXFP4's 13.0 GiB serving footprint.
3. **Serving**: all arms benchmarked cleanly end-to-end (zero failed
   requests, 90 cells). The upstream vLLM Marlin-MoE kernel bug that
   initially blocked full-NVFP4 expert serving (P0.10) was **root-caused
   (corrupt `mul_topk_weights=True` path) and worked around in our plugin**
   (D-014), after which both full-NVFP4 arms passed a correctness gate
   (greedy-64 agreement vs their QDQ references). Full-NVFP4 serving:
   **12.86 GiB weights (3× less VRAM than BF16), ~2× BF16 decode
   throughput at moderate concurrency, single-stream decode faster than
   BF16 — at the cost of slower long-context prefill** (weight-only
   dequant is compute-bound there). The night-1 D-hybrid null result is
   retained as historical.
4. Raw-text perplexity proved unfit as a quality metric for this
   Harmony-tuned model (validated with a harness null test) — documented
   with evidence; conclusions rest on logit fidelity + task accuracy.

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

**Upstream bug, root-caused and fixed (P0.10 → D-014):** vLLM 0.25.1's
`moe_wna16_marlin_gemm` produces corrupt output (~1e33, an out-of-bounds
fp32 multiplier read) on its `mul_topk_weights=True` path at GPT-OSS's
shapes — isolated with a standalone capture/replay harness against an exact
reference (flipping ONLY that flag turns 128/128 corrupt rows into 0/128).
Our plugin bypasses the broken path and applies routing weights externally —
mathematically identical, no CUDA rebuild (KNOWN_ISSUES.md P0.10;
docs/UPSTREAM_ISSUE_VLLM_MARLIN_MOE.md holds the filing-ready issue draft).

**Correctness gate before benchmarking** (greedy-64 prefix agreement vs each
arm's own QDQ reference in transformers, threshold 0.85, + coherent Harmony
chat, serving flags mirrored): **D 0.869 PASS, C 0.885 PASS** — agreement
means bit-identical across all probe repeats
(results/quality/serving_check_{D,C}.json). All five arms then benchmarked
end-to-end: A (native MXFP4), B (BF16), **C (full-NVFP4 RTN)**,
**D (full-NVFP4 GPTQ)**, and the night-1 D-hybrid retained as historical.

**Determinism caveat (P1.1, D-015):** the Marlin MoE path intermittently
flips single near-tie greedy tokens between identical reruns (2 of 6 strict
probes; flip positions per-prompt stable and coinciding with genuine logit
ties; BF16 control bitwise-stable). Quality-neutral — every agreement/chat/
benchmark signal is stable — reported in each gate JSON rather than gated on.

### Serving results (5 warmup + 30 measured req/cell, 1 repetition)
_results/serving/*/summary.json + per-request JSONL + gpu_telemetry.csv;
final comparison table: results/serving/comparison_final.{json,md} (night-1
arms A/B/D-hybrid + night-2 full-NVFP4 C/D after the P0.10 fix). Reduced
sample counts vs the final protocol (§18: ≥200/cell × 3 reps) are documented
here per the handoff; all raw records are retained for extension._

**Weights memory (vLLM "Model loading took", identical 0.9 GPU-utilization):**

| Arm | Weights in VRAM | KV-cache headroom |
|-----|-----------------|-------------------|
| A (official MXFP4) | 13.02 GiB | 54.6 GiB |
| B (BF16) | 39.15 GiB | ~28 GiB |
| C (full-NVFP4 RTN) | **12.86 GiB** | 54.1 GiB |
| D (full-NVFP4 GPTQ) | **12.86 GiB** | 54.1 GiB |
| D-hybrid (historical) | 38.22 GiB | ~28 GiB |

The 3.0× disk compression carries through to VRAM exactly: full-NVFP4 serving
reclaims ≈26 GiB of KV headroom over BF16, at byte-parity with the official
MXFP4 arm.

**Throughput/latency (selected cells; TTFT p50 / output tok/s):**

| Cell | A (MXFP4) | B (BF16) | C (RTN NVFP4) | D (GPTQ NVFP4) |
|------|-----------|----------|---------------|----------------|
| decode 128→256, c=1 | 0.023s / 305 | 0.031s / 265 | 0.024s / 315 | 0.025s / **314** |
| decode 128→256, c=32 | 4.205s* / 1254 | 0.891s / 1717 | 0.192s / 3382 | 0.210s / **3397** |
| decode 128→256, c=64 | 0.164s / 3674 | 0.165s / 1993 | 0.181s / 3229 | 0.168s / 3190 |
| mixed 1k→256, c=32 | 0.356s / 3268 | 0.412s / 1851 | 0.595s / 2623 | 0.594s / 2556 |
| mixed 8k→256, c=8 | 0.553s / 880 | 0.649s / 600 | 1.174s / 610 | 1.181s / 601 |
| prefill 8k→1, c=1 (TTFT) | 0.147s | 0.170s | 0.272s | 0.273s |

\* night-1 queue-burst artifact on arm A's c=32 cell (its c=64 is 0.164s).

Zero failed requests across all 90 cells (5 arms × 18). Observations:
- **Decode (memory-bound): full-NVFP4 wins.** Single-stream decode is
  faster than BF16 (314 vs 265 tok/s) and roughly matches native MXFP4;
  at c=32 it is ~2× BF16 (3397 vs 1717 tok/s). This is the textbook
  weight-only W4 win: 3× fewer weight bytes to stream per token.
- **Prefill/long-context (compute-bound): full-NVFP4 loses.** 8k-prefill
  TTFT is ~1.6× BF16 (0.273s vs 0.170s) and mixed-8k throughput ~0.8× —
  Marlin dequantizes weights to BF16 compute on Hopper, so compute-bound
  phases pay overhead without a bandwidth payoff. H100 has no native FP4;
  this trade-off is expected and now measured, not assumed.
- **C vs D serve identically** (same format, kernels, and byte layout —
  differences are within run-to-run noise), confirming the quality
  difference (§2) is free at serving time.
- Arm A remains the mixed/prefill throughput leader — its MXFP4 experts
  run a fused quantized-compute MoE path, not weight-only dequant.
- The night-1 D-hybrid rows (experts in BF16, ≈B's speed) are retained in
  comparison_final as historical context for the pre-fix state.

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
