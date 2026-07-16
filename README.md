# gpt-oss-20b Blockwise-GPTQ → NVFP4 on RunPod H100

Repairing and hardening a blockwise-GPTQ implementation to produce and
evaluate an **NVFP4 W4A16** quantization of `openai/gpt-oss-20b` on a single
**NVIDIA H100 80GB**, served with vLLM.

Note: the GitHub repo is named `gpt-oss-120b-blockwise-h100`, but the target
model is **gpt-oss-20b**.

## 1. Goal and non-goals

**Goal:** starting from the official MXFP4 checkpoint, produce (A) a pinned
official baseline, (B) a validated dequantized BF16 source, (C) a matched RTN
NVFP4 control, and (D) a blockwise-GPTQ NVFP4 treatment; validate correctness
at tensor/logit/task level; serve all arms with vLLM on the same H100 and
benchmark quality, memory, and serving performance.

**Non-goals:** AWS Trainium/Neuron, native Blackwell FP4 benchmarking, a full
MXFP4 GPTQ exporter, publishing checkpoints to HF, or claiming NVFP4 is the
optimal H100 format. NVFP4 on H100 runs through **weight-only Marlin
kernels** — memory savings are expected, speedups are not promised, and a
null/negative speed result is a valid outcome.

## 2. Hardware / environment

1× NVIDIA H100 80GB HBM3 (driver 580.126.09, CUDA 13.0), Python 3.12,
`/workspace` persistent. Environment lockfiles + system manifest: `envs/`.

Two isolated venvs are **mandatory** (their torch versions conflict):

```bash
scripts/bootstrap_quant_env.sh    # .venv-quant: torch 2.13, transformers 5.14
scripts/bootstrap_serve_env.sh    # .venv-serve: vllm 0.25.1 (torch 2.11)
scripts/capture_system_manifest.sh
# serving additionally requires the local vLLM patch plugin:
uv pip install --python .venv-serve/bin/python -e vllm-gptoss-nvfp4-plugin
```

## 3. Provenance (read before interpreting any result)

The official checkpoint stores expert weights in MXFP4. Arm B is its **exact
decode** to BF16 — *not* the unavailable pre-MXFP4 master. Everything
downstream is a **transquantization** experiment:

```
official MXFP4 (A) → dequant BF16 (B) → {RTN (C), blockwise-GPTQ (D)} NVFP4
```

Details + validation evidence: `docs/DEQUANTIZATION_PROVENANCE.md`,
`results/dequant_validation.json` (decode is bit-exact for every expert
tensor; all other tensors byte-identical).

## 4. The four arms

| ID | Path under `models/` | Role |
|----|----------------------|------|
| A | `gpt-oss-20b-official-mxfp4` | official deployment baseline |
| B | `gpt-oss-20b-mxfp4-dequant-bf16` | quantizer source |
| C | `gpt-oss-20b-mxfp4-dequant-rtn-nvfp4[-packed]` | matched RTN control |
| D | `gpt-oss-20b-mxfp4-dequant-blockwise-gptq-nvfp4[-packed]` | treatment |

C and D share the identical tensor mask, scale rules (D-010), exact-artifact
pipeline, exporter, and serving path — only the algorithm differs.
See `docs/EXPERIMENT_DESIGN.md`.

## 5. Repository architecture

See `docs/ARCHITECTURE.md`. Key invariants: memory-bounded resumable Hessian
collection (P0.4), exact code/scale capture verified bit-exact twice (P0.6),
complete per-tensor disposition manifest (P0.5), fail-closed everywhere, and
a source-verified vLLM packing contract (`docs/VLLM_NVFP4_CONTRACT.md`).

## 6. Commands (all verified on this pod)

Activate: quant work uses `.venv-quant/bin/python`; serving uses
`.venv-serve` (`scripts/serve_vllm.sh` handles PATH).

**Download + pin arm A** (idempotent, provenance manifest):
```bash
.venv-quant/bin/python scripts/download_official_model.py
```

**Dequantize + validate arm B:**
```bash
.venv-quant/bin/python scripts/dequantize_gpt_oss_20b.py
.venv-quant/bin/python scripts/validate_dequantized_source.py
```

**Unit/property tests** (CPU-safe, run from any cwd):
```bash
.venv-quant/bin/python -m pytest -q blockwise-gptq-main/tests/internalTests \
    --ignore=blockwise-gptq-main/tests/internalTests/test_vLLM_deploy_quantized_model.py
.venv-quant/bin/python blockwise-gptq-main/tests/stage1_nvfp4_unit_tests.py
.venv-quant/bin/python blockwise-gptq-main/tests/stage2_nvfp4_algorithm_tests.py
.venv-quant/bin/python blockwise-gptq-main/tests/stage3_gpt_oss_shape_tests.py
```

**Pilot (tiny end-to-end, §13 exit gates)** — must pass before any full run:
```bash
bash scripts/run_pilot.sh          # GPTQ 32×512 → pack → vLLM gates → RTN → A/B smoke
```

**Full quantization (arms D then C, resumable):**
```bash
bash scripts/run_full.sh
```

**Quality evaluation:**
```bash
# perplexity (diagnostic): baseline once, then per QDQ arm
.venv-quant/bin/python blockwise-gptq-main/tests/stage4_baseline_perplexity.py \
    --model_path models/gpt-oss-20b-mxfp4-dequant-bf16
.venv-quant/bin/python blockwise-gptq-main/tests/stage6_eval_perplexity.py \
    --model_path models/gpt-oss-20b-mxfp4-dequant-blockwise-gptq-nvfp4
# logit-level paired metrics (B reference)
.venv-quant/bin/python scripts/logit_eval.py \
    --reference B=models/gpt-oss-20b-mxfp4-dequant-bf16 \
    --candidates C=models/gpt-oss-20b-mxfp4-dequant-rtn-nvfp4 \
                 D=models/gpt-oss-20b-mxfp4-dequant-blockwise-gptq-nvfp4 \
    --out results/quality/logit_eval.json
# task-level suite (Harmony chat, greedy)
.venv-quant/bin/python scripts/task_eval.py --model <arm-path> --name <arm> \
    --out results/quality/task_<arm>.json
```

**Serving (one arm at a time, fresh server, identical flags):**
```bash
scripts/serve_vllm.sh configs/serve-gptq-nvfp4.env      # or the other 3 envs
# then, in another shell:
.venv-serve/bin/python scripts/serving_benchmark.py \
    --model gpt-oss-20b-mxfp4-dequant-blockwise-gptq-nvfp4 \
    --label final-rep1 --suites prefill decode mixed \
    --warmup 10 --requests 50 --reps 1
```

**Resume behavior:** stage 5 resumes from the per-layer Hessian cache
(SHA-256-verified manifest; calibration tokens are cached immutably and
hash-checked). Downloads and the dequantizer detect complete outputs and
skip. Interrupted serving benchmarks are per-cell JSONL — rerun the cell.

## 7. Outputs

- `models/` — checkpoints and packed artifacts (gitignored; each carries a
  provenance/packing manifest inside the directory)
- `results/` — validation, quality, and serving results (JSON/JSONL)
- `logs/` — timestamped logs for every stage
- `PROGRESS.md` / `DECISIONS.md` / `KNOWN_ISSUES.md` — the project record

## 8. Known limitations

- NVFP4 is non-native on H100: weight-only Marlin path (both linear and
  FusedMoE experts), confirmed from vLLM source and server logs.
- Arm B is a decoded-MXFP4 source (see §3) — quality deltas are measured
  against it, not against the original master.
- Serving arm A uses vLLM's native MXFP4 path (different kernels than C/D).
- Four vLLM 0.25.1 upstream gaps are patched by `vllm-gptoss-nvfp4-plugin` —
  including the Marlin MoE `mul_topk_weights` output-corruption bug that
  initially blocked full-NVFP4 serving (see `docs/VLLM_NVFP4_CONTRACT.md` §6,
  `docs/UPSTREAM_ISSUE_VLLM_MARLIN_MOE.md`, and `docs/TROUBLESHOOTING.md`).

## 9. Current results (complete)

**Headline:** blockwise GPTQ is 2.24× closer to the BF16 source than matched
RTN at identical NVFP4 format (KL 0.0113 vs 0.0254; top-1 1.00 for both;
task suite at ceiling for both). The full-NVFP4 artifact is 13 GB vs 39 GB
BF16 and serves in **12.86 GiB VRAM** with **~2× BF16 decode throughput** at
moderate concurrency (slower compute-bound prefill — expected for weight-only
W4 on Hopper). The upstream vLLM Marlin-MoE corruption (P0.10) was
root-caused and worked around in the plugin (DECISIONS D-014); all five
serving arms completed with zero failed requests.

Full write-up: `docs/REPORT.md`. Evidence trail: `PROGRESS.md`.
