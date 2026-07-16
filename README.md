# gpt-oss-20b Blockwise-GPTQ → NVFP4 on RunPod H100

Repairing and hardening a blockwise-GPTQ implementation to produce and evaluate an
**NVFP4 W4A16** quantization of `openai/gpt-oss-20b` on a single **NVIDIA H100 80GB**,
served with vLLM.

> **Status: setup / scaffolding phase.** The engineering work described in the agent
> handoff has not started yet. Do **not** treat the sub-repo's original "Quick Start"
> as a safe full-run path — several known P0 correctness/memory issues must be fixed
> first (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)). This README documents only what is
> currently true; run commands will be added as they are verified on this pod.

Note: the GitHub repo is named `gpt-oss-120b-blockwise-h100`, but the current target
model is **gpt-oss-20b**.

---

## 1. Goal and non-goals

**Goal:** use the existing `blockwise-gptq` source on one H100 to (1) preserve the
official MXFP4 checkpoint, (2) dequantize it to a clean BF16 source, (3) repair the
blockwise-GPTQ path, (4) produce a blockwise-GPTQ NVFP4 artifact and a matched RTN
NVFP4 control, (5) validate correctness at tensor/logit/task level, and (6) serve and
benchmark the arms on the same H100 with vLLM.

**Non-goals:** AWS Trainium/Neuron, native Blackwell FP4 benchmarking, a full MXFP4
GPTQ exporter, rewriting the repo before a pilot, or publishing checkpoints to HF
without explicit permission. NVFP4 is *not* claimed to be the optimal H100 format.

See [DECISIONS.md](DECISIONS.md) for the reasoning behind these scope choices.

---

## 2. Hardware and environment (this pod)

| | |
|---|---|
| GPU | 1× NVIDIA H100 80GB HBM3 (no native FP4 Tensor Cores) |
| Driver / CUDA | 580.126.09 / CUDA 13.0 |
| Host Python | 3.12.3 |
| RAM | ~2 TB |
| Persistent storage | `/workspace` (mfs, ample free space) |

H100 serves NVFP4 through a weight-only kernel (e.g. Marlin), subject to the pinned
vLLM version. A speedup is **not** assumed; a null/negative result is valid.

---

## 3. Source-model provenance

The official `openai/gpt-oss-20b` checkpoint stores expert weights in **MXFP4**. We
decode those to BF16 and use the result as the quantizer's source:

```
official MXFP4 checkpoint  →  exact MXFP4 decode  →  BF16 tensors (the "BF16 source")
```

This is **not** the original pre-MXFP4 master checkpoint, so the experiment is a
**transquantization**: official MXFP4 → dequantized BF16 → blockwise-GPTQ NVFP4.
Accurate names are used everywhere (e.g. `gpt-oss-20b-mxfp4-dequant-bf16`); the
dequantized source is never labeled "original BF16".

---

## 4. Experimental arms

| ID | Model | Purpose |
|----|-------|---------|
| A | Official `openai/gpt-oss-20b` MXFP4 | Real-world HF deployment baseline |
| B | Official MXFP4 decoded to BF16 | Exact source into the quantizer |
| C | RTN NVFP4 from B | Same format/packing path, no GPTQ (control) |
| D | Blockwise-GPTQ NVFP4 from B | Primary treatment |

Key comparisons: **D vs C** (pure GPTQ benefit at fixed format), **D vs B** (NVFP4
conversion cost), **D vs A** (custom artifact vs official), **A vs B** (validates the
dequantization). C and D must use identical tensor masks, block sizes, scale rules,
packing, and vLLM path — only the algorithm differs.

---

## 5. Repository layout

```
.
├── README.md                 # this file
├── PROGRESS.md               # chronological work log (proof required per entry)
├── DECISIONS.md              # non-obvious decisions + rationale
├── KNOWN_ISSUES.md           # open P0/P1/P2 issues from the handoff
├── H100_RUNPOD_..._HANDOFF.md# the full project specification / engineering contract
└── blockwise-gptq-main/      # the blockwise-GPTQ source being repaired
    ├── opteam-blockwise-gptq/#   core library (gptq, quantizer, expert dispatch, ...)
    ├── tests/                #   stageN_*.py pipeline + internalTests/ property tests
    ├── scripts/              #   download_model.sh, setup_runtime.sh
    ├── results/              #   prior run outputs (DeepSeek-V2-Lite examples)
    └── requirements.txt
```

The authoritative specification is
[H100_RUNPOD_GPT_OSS_20B_BLOCKWISE_GPTQ_AGENT_HANDOFF.md](H100_RUNPOD_GPT_OSS_20B_BLOCKWISE_GPTQ_AGENT_HANDOFF.md).
For details on the sub-repo's stages and formats, see
[blockwise-gptq-main/README.md](blockwise-gptq-main/README.md).

---

## 6. Planned workflow (not yet runnable)

Per the handoff, work proceeds in this order — each step gated on the previous one:

```
static audit → env bootstrap → focused unit tests → dequantization validation
→ tiny end-to-end pilot → packed-load validation → RTN/GPTQ matched pilot
→ full calibration → final serving benchmarks
```

Two isolated Python environments will be created (quant deps and vLLM deps conflict):

- `.venv-quant` — Transformers load + MXFP4 dequant, calibration, blockwise GPTQ,
  QDQ validation, export, tests.
- `.venv-serve` — vLLM serving, OpenAI-compatible API tests, async benchmarks.

Verified activation/run commands will be added here as each stage is validated on the
pod. Until then, see the handoff for the intended commands.

---

## 7. Progress and issues

- Current status and history: [PROGRESS.md](PROGRESS.md)
- Open blockers and risks: [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
- Decisions and rationale: [DECISIONS.md](DECISIONS.md)

---

## 8. Known limitations

- NVFP4 is **non-native** on H100; expect a weight-only kernel, not FP4 Tensor Cores.
- The "BF16 source" is a dequantized MXFP4 checkpoint, not an original master (see §3).
- The source pipeline has documented P0 correctness/memory issues still to be fixed
  before any full run — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
