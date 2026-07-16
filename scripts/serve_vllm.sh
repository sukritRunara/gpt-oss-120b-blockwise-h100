#!/usr/bin/env bash
# Parameterized vLLM server launcher (handoff §17).
#
# Usage:
#   scripts/serve_vllm.sh configs/serve-gptq-nvfp4.env [extra vllm args...]
#
# The env file must set MODEL_PATH and SERVED_NAME; optional PORT (default
# 8000), MAX_MODEL_LEN (default 16384), EXTRA_ARGS. Everything else is kept
# identical across arms (handoff: only the model path and strictly required
# flags may differ). Prefix caching is DISABLED for benchmark comparability.
#
# Logs: logs/serving/<SERVED_NAME>_<UTC>.log  (full stdout/stderr + the exact
# command line, model-load time and VRAM are in the vLLM log itself).
set -euo pipefail

ENV_FILE="$1"; shift || true
# shellcheck disable=SC1090
source "$ENV_FILE"

PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs/serving"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/${SERVED_NAME}_${STAMP}.log"

export PATH="$ROOT/.venv-serve/bin:$PATH"

CMD=("$ROOT/.venv-serve/bin/vllm" serve "$MODEL_PATH"
     --served-model-name "$SERVED_NAME"
     --port "$PORT"
     --max-model-len "$MAX_MODEL_LEN"
     --gpu-memory-utilization 0.90
     --no-enable-prefix-caching
     --disable-log-requests
     ${EXTRA_ARGS:-}
     "$@")

{
  echo "# launched: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# command: ${CMD[*]}"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
} | tee "$LOG"

exec "${CMD[@]}" >> "$LOG" 2>&1
