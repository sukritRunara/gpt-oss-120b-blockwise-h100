# Experiment design

## Question

Does blockwise GPTQ improve NVFP4 W4A16 quantization quality over RTN for
GPT-OSS-20B at matched format/packing/serving, and what are the memory and
serving-performance consequences of NVFP4 on an H100 (non-native FP4)?

## Arms

| ID | Artifact | Role |
|----|----------|------|
| A | `gpt-oss-20b-official-mxfp4` | real-world deployment baseline |
| B | `gpt-oss-20b-mxfp4-dequant-bf16` | exact decode of A; quantizer source |
| C | `gpt-oss-20b-mxfp4-dequant-rtn-nvfp4` | matched RTN control |
| D | `gpt-oss-20b-mxfp4-dequant-blockwise-gptq-nvfp4` | treatment |

Valid interpretations: D-vs-C = GPTQ algorithm effect at fixed format;
D-vs-B = cost of NVFP4 conversion; D-vs-A = practical artifact vs official;
A-vs-B = decode validation (completed: bit-exact decode, byte-identical
passthrough — results/dequant_validation.json).

C/D matching guarantees: same source (B), same E2M1 grid + fp8 block scales +
D-010 global-scale rule (incl. shared q/k/v), same exact-artifact capture and
verification, same manifest schema, same Stage-7 exporter and vLLM plugin,
same serve flags. `build_rtn_control.py --match_manifest` replicates D's
tensor mask exactly. The ONLY difference is the algorithm.

## Configurations

Pilot (§13): C4, 32 samples × 512 tokens, Hessian group size 1, GPTQ block
128, NVFP4 microblock 16, percdamp 0.01, seed 0, no mixed-precision fallback.

Full run (§14): C4, 512 samples × 2048 tokens, otherwise identical. Any
change is recorded in DECISIONS.md before the run.

## Quality evaluation (kept separate from performance)

1. Tensor level: dequantize(packed) == QDQ bit-exact (enforced twice in the
   pipeline, fail-closed).
2. Logit level: fixed held-out prompt set, arms B/C/D under identical inputs
   (transformers, GPU): cosine, KL, top-1/top-k agreement, max abs diff,
   greedy prefix agreement.
3. Perplexity (diagnostic only): stage 4/6 on identical cached token
   sequences, WikiText-2 + C4.
4. Task level: small reproducible suite over general knowledge / math / code
   / instruction following with Harmony chat formatting; every prompt, raw
   output, and score saved.

Calibration data (C4 train shard) is disjoint from evaluation data
(WikiText-2 test-style slices, held-out prompts); no tuning against the
evaluation sets.

## Serving evaluation

vLLM 0.25.1, one H100, TP=1, identical launch flags per arm (configs/
serve-*.env), prefix caching disabled, fresh server per arm, full logs kept.
Suites and metrics per handoff §18 via scripts/serving_benchmark.py: prefill
(1k/8k×1), decode (128×256), mixed (1k/8k×256), 32k cells explicitly
opt-in after a capacity check; concurrency sweeps; ≥3 repetitions for final
numbers with alternating arm order; JSONL per request; GPU telemetry sampled.

Known interpretation caveats:
- NVFP4 on H100 uses weight-only Marlin kernels: memory savings are expected,
  speedups are NOT promised — a null/negative result is valid.
- Arm A serves through vLLM's native gpt-oss MXFP4 path (different kernels).
