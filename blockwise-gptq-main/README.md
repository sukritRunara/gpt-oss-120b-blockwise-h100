# Block-wise GPTQ Quantization Pipeline

A modular, architecture-agnostic pipeline for quantizing large language models using GPTQ with block-wise weight quantization. Supports multiple model architectures (dense and MoE) and six quantization precision formats.

---

## Overview

The pipeline converts a BF16 model to a quantized format (default: NVFP4) using GPTQ — a post-training quantization method that minimizes per-layer reconstruction error using second-order Hessian information. A parallel Hessian collection mode prevents the layer-by-layer error cascade that occurs in sequential quantization.

```
BF16 model
   │
   ├─ Stage 4: Measure BF16 perplexity baseline (WikiText-2 + C4)
   │
   ├─ Stage 5: Quantize → <model>-<FORMAT>/
   │               ├── Parallel Hessian collection (all layers, BF16 weights)
   │               └── Block-wise GPTQ per layer
   │
   └─ Stage 6: Measure quantized perplexity → Δppl vs BF16
```

Stages 1–3 are validation/test stages that verify correctness before running on real hardware.

---

## Supported Models

| Model | Architecture | Expert type | Handler |
|---|---|---|---|
| GPT-OSS 20B | MoE (Mixture of Experts) | Batched tensor `[E, in, out]` | `GptOssHandler` |
| DeepSeek V2 Lite | MoE | Routed `nn.Linear` | `DeepSeekV2Handler` |
| Any dense HuggingFace model | Dense | Standard `nn.Linear` | _(no handler needed)_ |

Adding a new architecture requires implementing a `MoEHandler` subclass in `expert_dispatch.py` and registering it in `_HANDLER_REGISTRY`.

---

## Supported Quantization Formats

| Flag | Format | Description |
|---|---|---|
| `nvfp4` _(default)_ | NVIDIA FP4 | Block-wise, block_size=16. Native format for Blackwell GPUs. |
| `mxint4` | MX INT4 | Block-wise, block_size=16. MX (Microscaling) format. |
| `int4` | INT4 symmetric | Group quantization, groupsize=128. |
| `int4_perchannel` | INT4 symmetric | Per-channel (no grouping). |
| `int8` | INT8 symmetric | Per-channel. |
| `fp8` | FP8 E4M3 | Per-channel. |

Pass `--quant_format <format>` to Stage 5 to select the format.

---

## Prerequisites
**Hardware:** NVIDIA GPU (Blackwell/H100/A100 recommended for NVFP4). Multi-GPU supported via `device_map="auto"`.

**Software:** Run `setup_runtime.sh` to create the environment.

**Model weights:** Download the BF16 model weights to `models/<model-name>/` before running. You can use `download_model.sh` to download a model — edit the script to set the model you want.

---

## Pipeline Stages

### Stage 1 — NVFP4 Unit Tests
Verifies that `NVFP4Quantizer` produces correctly scaled, correctly shaped FP4 outputs on synthetic tensors. No GPU required.

```bash
python tests/stage1_nvfp4_unit_tests.py
```

### Stage 2 — GPTQ Algorithm Tests
Verifies that the GPTQ reconstruction loss is lower than RTN (round-to-nearest) and that Hessian accumulation is correct.

```bash
python tests/stage2_gptq_algorithm_tests.py
```

### Stage 3 — Shape Tests

Verifies that the full quantization pipeline produces correct weight shapes and that expert write-back is correct for each supported architecture.

```bash
# GPT-OSS 20B (MoE with batched expert tensors)
python tests/stage3_gpt_oss_shape_tests.py

# DeepSeek V2 Lite (MoE with routed nn.Linear experts)
python tests/stage3_deepseek_v2_lite_shape_tests.py
```

All shape tests run on CPU with tiny synthetic models — no GPU or real weights needed.

### Stage 4 — Baseline Perplexity

Loads the BF16 model and records perplexity on WikiText-2 and C4. Also caches the evaluation samples so Stage 6 uses the exact same tokens.

```bash
# GPT-OSS 20B
python tests/stage4_baseline_perplexity.py \
    --model_path models/gpt-oss-20b-BF16

# DeepSeek V2 Lite (use seq_len 4096 — trained on longer context)
python tests/stage4_baseline_perplexity.py \
    --model_path models/DeepSeek-V2-Lite \
    --seq_len 4096

# C4 only (skip WikiText-2)
python tests/stage4_baseline_perplexity.py \
    --model_path models/<name> --skip_wikitext2
```

**Output:** `results/stage4_<model-name>_baseline.json`

> **Note:** Perplexity numbers are not comparable across models. A GPT-OSS 20B score of ~27 on WikiText-2 and a DeepSeek V2 Lite score of ~7 are both expected — they reflect different tokenizers, training data distributions, and model purposes. What matters is the **Δppl in Stage 6**, not the absolute value.

### Stage 5 — Quantize Model
Runs block-wise GPTQ quantization on the BF16 model. Parallel Hessian collection (default ON) collects all layer Hessians from the unmodified BF16 model before quantizing any layer, preventing the layer-by-layer error cascade.
```bash
# Default (NVFP4, C4 calibration, 512 samples)
python tests/stage5_quantize_model.py \
    --model_path models/gpt-oss-20b-BF16

# Different quantization format
python tests/stage5_quantize_model.py \
    --model_path models/gpt-oss-20b-BF16 \
    --quant_format int8

# DeepSeek V2 Lite, longer sequence for calibration
python tests/stage5_quantize_model.py \
    --model_path models/DeepSeek-V2-Lite \
    --seq_len 4096

# Custom output directory and results path
python tests/stage5_quantize_model.py \
    --model_path models/gpt-oss-20b-BF16 \
    --output_dir models/gpt-oss-20b-NVFP4-custom \
    --results results/my_run.json

# Optional blocksize sweep (tries B ∈ {64, 128, 256} on two probe layers first)
python tests/stage5_quantize_model.py \
    --model_path models/<name> --blocksize_search

# Quantize everything — disable mixed-precision BF16 fallback
python tests/stage5_quantize_model.py \
    --model_path models/<name> --mixed_precision_threshold 0

# Sequential mode (for debugging only — cascade-prone)
python tests/stage5_quantize_model.py \
    --model_path models/<name> --no_parallel_hessian
```
**Key parameters:**
| Parameter | Default | Notes |
|---|---|---|
| `--quant_format` | `nvfp4` | See format table above |
| `--blocksize` | `128` | GPTQ column block width; must be multiple of 16 |
| `--percdamp` | `0.01` | Hessian damping (larger = more stable, slightly worse quality) |
| `--n_calib` | `512` | Number of calibration samples |
| `--seq_len` | `2048` | Calibration sequence length (use 4096 for DeepSeek V2) |
| `--dataset` | `c4` | Calibration dataset: `c4` or `wikitext2` |
| `--mixed_precision_threshold` | `100.0` | Layers with GPTQ loss above this stay in BF16; set to `0` to quantize all |
| `--output_dir` | `models/<stem>-<FORMAT>` | Override the default output path |
| `--results` | `results/stage5_<stem>_<format>_quantize.json` | Override the default results path |
| `--blocksize_search` | off | Sweep B ∈ {64, 128, 256} on two probe layers and auto-select best |
| `--no_parallel_hessian` | off | Use sequential mode (debugging only — cascade-prone) |

**Output:**
- Quantized weights: `models/<model-name>-<FORMAT>/`
- Results JSON: `results/stage5_<model-name>_<format>_quantize.json`

### Stage 6 — Evaluate Quantized Model

Loads the quantized model and compares perplexity against the Stage 4 BF16 baseline using the same cached samples.

```bash
# Automatically finds baseline and sample caches from folder name
python tests/stage6_eval_perplexity.py \
    --model_path models/gpt-oss-20b-BF16-NVFP4

# DeepSeek V2 Lite
python tests/stage6_eval_perplexity.py \
    --model_path models/DeepSeek-V2-Lite-NVFP4 \
    --seq_len 4096

# C4 only
python tests/stage6_eval_perplexity.py \
    --model_path models/<name>-NVFP4 --skip_wikitext2
```

**Output:** `results/stage6_<model-name>-<FORMAT>_eval.json`

**Expected Δppl:**

| Dataset | Good | Warning |
|---|---|---|
| WikiText-2 | Δppl < 1.0 | Δppl > 1.5 |
| C4 | Δppl < 1.5 | Δppl > 2.0 |

---

## Quick Start — Full Run (GPT-OSS 20B, NVFP4)

```bash
# 1. Verify correctness (CPU, ~1 min)
python tests/stage1_nvfp4_unit_tests.py
python tests/stage2_gptq_algorithm_tests.py
python tests/stage3_gpt_oss_shape_tests.py

# 2. Baseline (GPU, ~10 min)
python tests/stage4_baseline_perplexity.py --model_path models/gpt-oss-20b-BF16

# 3. Quantize (GPU, ~60-90 min)
python tests/stage5_quantize_model.py --model_path models/gpt-oss-20b-BF16

# 4. Evaluate (GPU, ~10 min)
python tests/stage6_eval_perplexity.py --model_path models/gpt-oss-20b-BF16-NVFP4
```

## Quick Start — DeepSeek V2 Lite, INT8

```bash
python tests/stage3_deepseek_v2_lite_shape_tests.py

python tests/stage4_baseline_perplexity.py \
    --model_path models/DeepSeek-V2-Lite --seq_len 4096

python tests/stage5_quantize_model.py \
    --model_path models/DeepSeek-V2-Lite \
    --quant_format int8 --seq_len 4096

python tests/stage6_eval_perplexity.py \
    --model_path models/DeepSeek-V2-Lite-INT8 --seq_len 4096
```

---

## Results Interpretation

After Stage 6 completes, open `results/stage6_<name>_eval.json`:

```json
{
  "wikitext2": {
    "ppl_bf16":  7.2240,
    "ppl_nvfp4": 7.3812,
    "delta_ppl": +0.1572
  },
  "c4": {
    "ppl_bf16":  11.409,
    "ppl_nvfp4": 11.591,
    "delta_ppl": +0.182
  }
}
```

A Δppl under ~0.5 on both datasets indicates excellent quantization quality. Values above the warning thresholds (1.5 / 2.0) suggest investigating `--percdamp`, `--blocksize`, or `--n_calib`.

---

edit. The true structure is as follow:

1. opteam-blockwise-gptq/
2. results/
3. scripts/ downlaod_model.sh and setup_runtime.sh
4. tests/ also include a folder name internalTests with test_property1_gtpq_beats_rtn.py, test_property2_hessian_accumulaiotn and test_property4_expeert_writeback
The is also a readme and requirmenrt,txt fiel


## Code Structure
```
blockwise-gptq/
├── README.md
├── requirements.txt
├── opteam-blockwise-gptq/
│   ├── apply.py                          # Core GPTQ orchestration (layer loop, Hessian collection)
│   ├── expert_dispatch.py                # MoEHandler registry (GPT-OSS, DeepSeek V2)
│   ├── deepseek_v2_lite_expert_gptq.py
│   ├── gpt_oss_expert_gptq.py
│   ├── calibration.py                    # Calibration data loading and LayerInputCatcher
│   ├── model_utils.py                    # get_model_layers(), find_layers(), get_embedding_layers()
│   ├── gptq.py                           # GPTQ class (Hessian accumulation, fasterquant_blockwise)
│   └── quantizer.py                      # QUANTIZER_REGISTRY + all quantizer classes
├── results/
├── scripts/
│   ├── download_model.sh
│   └── setup_runtime.sh
└── tests/
    ├── stage1_nvfp4_unit_tests.py
    ├── stage2_gptq_algorithm_tests.py
    ├── stage3_gpt_oss_shape_tests.py
    ├── stage3_deepseek_v2_lite_shape_tests.py
    ├── stage4_baseline_perplexity.py
    ├── stage5_quantize_model.py
    ├── stage6_eval_perplexity.py
    └── internalTests/
        ├── test_property1_gptq_beats_rtn.py
        ├── test_property2_hessian_accumulation.py
        └── test_property4_expert_writeback.py
```
