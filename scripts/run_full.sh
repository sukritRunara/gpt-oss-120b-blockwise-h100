#!/usr/bin/env bash
# FULL quantization run (handoff §14). LAUNCH ONLY AFTER THE PILOT GATES PASS.
#
#   C4, 512 samples × 2048 tokens, Hessian group size 1, GPTQ block 128,
#   NVFP4 microblock 16, percdamp 0.01, seed 0, no mixed-precision fallback.
#
# Fully resumable: the Hessian cache (per-layer, SHA-256 manifest) and the
# immutable token cache mean an interrupted run continues where it stopped.
# Produces arm D (QDQ + packed) and arm C (matched RTN, QDQ + packed).
set -euo pipefail

ROOT=/workspace
REPO=$ROOT/blockwise-gptq-main
QP=$ROOT/.venv-quant/bin/python
LOGS=$ROOT/logs/quantization
RES=$ROOT/results
mkdir -p "$LOGS" "$RES/quality"

SRC=$ROOT/models/gpt-oss-20b-mxfp4-dequant-bf16
D_QDQ=$ROOT/models/gpt-oss-20b-mxfp4-dequant-blockwise-gptq-nvfp4
D_PACKED=${D_QDQ}-packed
C_QDQ=$ROOT/models/gpt-oss-20b-mxfp4-dequant-rtn-nvfp4

export HF_HOME=$ROOT/cache/huggingface
export HF_DATASETS_CACHE=$ROOT/cache/datasets

echo "=== [full 1/3] Stage 5: blockwise GPTQ (512×2048, group=1) ==="
$QP $REPO/tests/stage5_quantize_model.py \
    --model_path "$SRC" \
    --output_dir "$D_QDQ" \
    --dataset c4 --n_calib 512 --seq_len 2048 \
    --blocksize 128 --percdamp 0.01 \
    --mixed_precision_threshold 0 \
    --hessian_cache_dir "$ROOT/cache/hessians-full" \
    --hessian_layer_group_size 4 \
    --results "$RES/stage5_full.json" \
    2>&1 | tee "$LOGS/full_stage5.log" | grep -E "^\[GPTQ\]|Manifest|Total|Saved|Results" | tail -30

echo "=== [full 2/3] Stage 7: pack arm D ==="
$QP $REPO/tests/stage7_save_modelopt.py \
    --model_path "$D_QDQ" --output_dir "$D_PACKED" \
    2>&1 | tee "$LOGS/full_stage7.log" | tail -6

echo "=== [full 3/3] Matched RTN control (arm C) + pack ==="
$QP $ROOT/scripts/build_rtn_control.py \
    --source "$SRC" --output "$C_QDQ" \
    --match_manifest "$D_QDQ/quant_artifacts/manifest.json" \
    --pack \
    2>&1 | tee "$LOGS/full_rtn.log" | tail -4

echo "=== FULL RUN COMPLETE ==="
echo "arm D: $D_PACKED"
echo "arm C: ${C_QDQ}-packed"
