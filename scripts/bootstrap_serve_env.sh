#!/usr/bin/env bash
# Bootstrap the serving environment (.venv-serve) with uv.
#
# Idempotent: re-running installs into the existing venv. Delete
# /workspace/.venv-serve to force a clean rebuild.
#
# On success, freezes the resolved versions into
#   envs/serve-requirements.lock.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-serve"

echo "[bootstrap-serve] venv: $VENV"
uv venv --allow-existing "$VENV"

if [[ -s "$ROOT/envs/serve-requirements.lock.txt" ]]; then
    echo "[bootstrap-serve] Installing from lockfile (reproducible)"
    uv pip install --python "$VENV/bin/python" -r "$ROOT/envs/serve-requirements.lock.txt"
else
    echo "[bootstrap-serve] No lockfile — resolving from serve-requirements.in"
    uv pip install --python "$VENV/bin/python" -r "$ROOT/envs/serve-requirements.in"
    uv pip freeze --python "$VENV/bin/python" > "$ROOT/envs/serve-requirements.lock.txt"
    echo "[bootstrap-serve] Lockfile written: envs/serve-requirements.lock.txt"
fi

"$VENV/bin/python" - <<'EOF'
import vllm, torch
print(f"[bootstrap-serve] OK  vllm={vllm.__version__}  torch={torch.__version__}  "
      f"cuda_available={torch.cuda.is_available()}")
EOF
echo "[bootstrap-serve] Done. Activate with: source $VENV/bin/activate"
