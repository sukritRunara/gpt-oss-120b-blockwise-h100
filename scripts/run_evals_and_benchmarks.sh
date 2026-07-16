#!/usr/bin/env bash
# Post-full-run pipeline: quality evals (B/C/D QDQ) + D-hybrid pack +
# serving benchmarks (A, B, D-hybrid). Sequential — one GPU.
set -euo pipefail

ROOT=/workspace
REPO=$ROOT/blockwise-gptq-main
QP=$ROOT/.venv-quant/bin/python
LOGS=$ROOT/logs
RES=$ROOT/results
mkdir -p "$RES/quality" "$RES/serving" "$LOGS/benchmarks"

B=$ROOT/models/gpt-oss-20b-mxfp4-dequant-bf16
C=$ROOT/models/gpt-oss-20b-mxfp4-dequant-rtn-nvfp4
D=$ROOT/models/gpt-oss-20b-mxfp4-dequant-blockwise-gptq-nvfp4

export HF_HOME=$ROOT/cache/huggingface
export HF_DATASETS_CACHE=$ROOT/cache/datasets

echo "=== [eval 1/6] Perplexity baseline (B) ==="
$QP $REPO/tests/stage4_baseline_perplexity.py --model_path "$B" \
    2>&1 | tee "$LOGS/benchmarks/stage4_B.log" | tail -6

echo "=== [eval 2/6] Perplexity C and D (same cached tokens) ==="
$QP $REPO/tests/stage6_eval_perplexity.py --model_path "$D" \
    2>&1 | tee "$LOGS/benchmarks/stage6_D.log" | tail -6
$QP $REPO/tests/stage6_eval_perplexity.py --model_path "$C" \
    2>&1 | tee "$LOGS/benchmarks/stage6_C.log" | tail -6

echo "=== [eval 3/6] Logit-level paired metrics (B ref, C and D) ==="
$QP $ROOT/scripts/logit_eval.py \
    --reference B="$B" --candidates C="$C" D="$D" \
    --out "$RES/quality/logit_eval.json" \
    2>&1 | tee "$LOGS/benchmarks/logit_eval.log" | tail -5

echo "=== [eval 4/6] Task suite (B, C, D) ==="
for arm in B C D; do
    eval "path=\$$arm"
    $QP $ROOT/scripts/task_eval.py --model "$path" --name "$arm" \
        --out "$RES/quality/task_$arm.json" \
        2>&1 | tee "$LOGS/benchmarks/task_$arm.log" | tail -2
done

echo "=== [eval 5/6] Build D-hybrid serving artifact ==="
$QP - <<EOF
import importlib.util, json
from pathlib import Path
qdq = Path("$D")
man = json.load(open(qdq / "quant_artifacts" / "manifest.json"))
for r in man["tensors"]:
    if r["kind"] == "expert_slice":
        r["disposition"] = "BF16_FALLBACK"
        r["reason"] = "serving workaround for upstream vLLM Marlin NVFP4-MoE bug (P0.10)"
        r["artifact"] = None
iso = qdq / "quant_artifacts" / "manifest_hybrid.json"
iso.write_text(json.dumps(man))
spec = importlib.util.spec_from_file_location("stage7", "$REPO/tests/stage7_save_modelopt.py")
stage7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage7)
report = stage7.pack_from_manifest(qdq, Path("${D}-hybrid-linears-nvfp4-experts-bf16"),
                                   iso, allow_hybrid=True)
print("hybrid pack:", report["counts"], f"bf16_frac={report['bf16_fraction']:.3f}")
EOF

cat > $ROOT/configs/serve-gptq-hybrid.env <<CFG
MODEL_PATH=${D}-hybrid-linears-nvfp4-experts-bf16
SERVED_NAME=gpt-oss-20b-gptq-nvfp4-HYBRID-experts-bf16
CFG

echo "=== [eval 6/6] Serving benchmarks: A, B, D-hybrid ==="
for cfg in serve-official-mxfp4 serve-dequant-bf16 serve-gptq-hybrid; do
    echo "--- arm: $cfg ---"
    bash $ROOT/scripts/run_serving_benchmarks.sh "$ROOT/configs/$cfg.env" night1 1 30 \
        2>&1 | tee "$LOGS/benchmarks/bench_$cfg.log" | grep -E "\[cell\]|ok=|ready|Summary|DIED" || true
done

echo "=== EVALS AND BENCHMARKS COMPLETE ==="
