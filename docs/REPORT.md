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

### Perplexity (diagnostic; identical cached token sequences)
_results/…stage4/stage6 JSONs_

| Arm | WikiText-2 | Δ vs B | C4 | Δ vs B |
|-----|-----------|--------|----|--------|
| B (BF16) | TBD | — | TBD | — |
| C (RTN) | TBD | TBD | TBD | TBD |
| D (GPTQ) | TBD | TBD | TBD | TBD |

### Logit-level paired metrics (24 held-out prompts; B is reference)
_results/quality/logit_eval.json_

| Metric | C (RTN) | D (GPTQ) |
|--------|---------|----------|
| cosine (mean / min) | TBD | TBD |
| KL divergence (mean) | TBD | TBD |
| top-1 agreement | TBD | TBD |
| greedy-64 prefix agreement | TBD | TBD |

### Task suite (40 items: knowledge/math/code/instruction, Harmony chat, greedy)
_results/quality/task_{B,C,D}.json_

| Arm | Overall | Knowledge | Math | Code | Instruct |
|-----|---------|-----------|------|------|----------|
| B | TBD | | | | |
| C | TBD | | | | |
| D | TBD | | | | |

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
