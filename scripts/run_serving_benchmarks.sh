#!/usr/bin/env bash
# Serve + benchmark each arm sequentially (handoff §17/§18).
#
#   bash scripts/run_serving_benchmarks.sh <config.env> <label> [reps] [requests]
#
# Starts a fresh vLLM server from the given env config, waits for readiness,
# runs the async benchmark suites, captures full logs + GPU telemetry, then
# shuts the server down cleanly. One arm per invocation; alternate arm order
# across repetitions manually to reduce time-order bias.
set -euo pipefail

ROOT=/workspace
CFG="$1"; LABEL="$2"; REPS="${3:-1}"; REQUESTS="${4:-30}"
# shellcheck disable=SC1090
source "$CFG"
PORT="${PORT:-8000}"

export PATH="$ROOT/.venv-serve/bin:$PATH"

echo "[bench] starting server for $SERVED_NAME"
bash "$ROOT/scripts/serve_vllm.sh" "$CFG" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

# Wait for readiness (up to 15 min for big model loads)
for i in $(seq 1 180); do
  if curl -s "http://localhost:$PORT/v1/models" | grep -q "$SERVED_NAME"; then
    echo "[bench] server ready after ~$((i * 5))s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[bench] SERVER DIED — see logs/serving/"
    exit 1
  fi
  sleep 5
done

# Deterministic smoke before benchmarking
curl -s "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$SERVED_NAME\", \"prompt\": \"The capital of France is\", \"max_tokens\": 8, \"temperature\": 0}" \
  | tee "$ROOT/results/serving/smoke_${SERVED_NAME}_${LABEL}.json"
echo

"$ROOT/.venv-serve/bin/python" "$ROOT/scripts/serving_benchmark.py" \
    --base_url "http://localhost:$PORT" \
    --model "$SERVED_NAME" \
    --label "$LABEL" \
    --suites prefill decode mixed \
    --warmup 5 --requests "$REQUESTS" --reps "$REPS"

echo "[bench] done — stopping server"
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
trap - EXIT
sleep 10
