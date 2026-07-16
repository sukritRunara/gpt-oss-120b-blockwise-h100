#!/usr/bin/env bash
# Tiny end-to-end pilot (handoff §13). Runs the COMPLETE path on small
# calibration settings and checks the exit gates. Full-run launch is gated
# on this script exiting 0.
#
#   dequant BF16 source → grouped Hessian cache → blockwise GPTQ (capture)
#   → QDQ + manifest → packed NVFP4 → vLLM load + determinism + Harmony chat
#   → QDQ-vs-packed serving agreement → matched RTN control (same mask)
#
# Pilot config (handoff §13): 32 samples × 512 tokens, group size 1,
# GPTQ block 128, NVFP4 microblock 16, percdamp 0.01, fixed seed,
# no mixed-precision fallback (full NVFP4 preferred for the primary claim).
set -euo pipefail

ROOT=/workspace
REPO=$ROOT/blockwise-gptq-main
QP=$ROOT/.venv-quant/bin/python
SP=$ROOT/.venv-serve/bin/python
LOGS=$ROOT/logs/quantization
RES=$ROOT/results/pilot
mkdir -p "$LOGS" "$RES"

SRC=$ROOT/models/gpt-oss-20b-mxfp4-dequant-bf16
GPTQ_QDQ=$ROOT/models/pilot-gptq-nvfp4
GPTQ_PACKED=$ROOT/models/pilot-gptq-nvfp4-packed
RTN_QDQ=$ROOT/models/pilot-rtn-nvfp4

export HF_HOME=$ROOT/cache/huggingface
export HF_DATASETS_CACHE=$ROOT/cache/datasets
export PATH="$ROOT/.venv-serve/bin:$PATH"

echo "=== [pilot 1/6] Stage 5: blockwise GPTQ (32×512, group=1) ==="
$QP $REPO/tests/stage5_quantize_model.py \
    --model_path "$SRC" \
    --output_dir "$GPTQ_QDQ" \
    --dataset c4 --n_calib 32 --seq_len 512 \
    --blocksize 128 --percdamp 0.01 \
    --mixed_precision_threshold 0 \
    --hessian_cache_dir "$ROOT/cache/hessians-pilot" \
    --hessian_layer_group_size 1 \
    --results "$RES/stage5_pilot.json" \
    2>&1 | tee "$LOGS/pilot_stage5.log" | grep -E "^\[GPTQ\]|Layer|group|Manifest|Total|Saved|Results" | tail -40

echo "=== [pilot 2/6] Stage 7: pack exact artifacts ==="
$QP $REPO/tests/stage7_save_modelopt.py \
    --model_path "$GPTQ_QDQ" --output_dir "$GPTQ_PACKED" \
    2>&1 | tee "$LOGS/pilot_stage7.log" | tail -8

echo "=== [pilot 3/6] QDQ greedy reference (transformers) ==="
$QP $ROOT/scripts/make_qdq_reference.py \
    --model "$GPTQ_QDQ" --out "$RES/qdq_reference.json" \
    2>&1 | tee "$LOGS/pilot_reference.log" | tail -4

echo "=== [pilot 4/6] vLLM serving gates (load/determinism/harmony/agreement) ==="
$SP $ROOT/scripts/pilot_serving_check.py \
    --packed "$GPTQ_PACKED" \
    --reference_json "$RES/qdq_reference.json" \
    --out "$RES/serving_check.json" \
    2>&1 | tee "$LOGS/pilot_serving.log" | tail -12

echo "=== [pilot 5/6] Matched RTN control (same mask) + pack ==="
$QP $ROOT/scripts/build_rtn_control.py \
    --source "$SRC" --output "$RTN_QDQ" \
    --match_manifest "$GPTQ_QDQ/quant_artifacts/manifest.json" \
    --pack \
    2>&1 | tee "$LOGS/pilot_rtn.log" | tail -6

echo "=== [pilot 6/6] Arm A + B vLLM smoke ==="
$SP - <<'EOF' 2>&1 | tee "$LOGS/pilot_ab_smoke.log" | tail -6
from vllm import LLM, SamplingParams
for path in ("/workspace/models/gpt-oss-20b-official-mxfp4",
             "/workspace/models/gpt-oss-20b-mxfp4-dequant-bf16"):
    print(f"--- smoke: {path}")
    llm = LLM(model=path, max_model_len=2048, gpu_memory_utilization=0.85,
              enforce_eager=True, disable_log_stats=True)
    out = llm.generate([{"prompt_token_ids": [200006, 1428, 200008, 400, 500]}],
                       SamplingParams(temperature=0.0, max_tokens=8))
    print("tokens:", list(out[0].outputs[0].token_ids))
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
print("AB_SMOKE_OK")
EOF

echo "=== PILOT COMPLETE — inspect $RES for gate evidence ==="
