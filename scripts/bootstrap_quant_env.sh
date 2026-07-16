#!/usr/bin/env bash
# Bootstrap the quantization environment (.venv-quant) with uv.
#
# Idempotent: re-running installs into the existing venv (uv resolves no-ops
# quickly). Delete /workspace/.venv-quant to force a clean rebuild.
#
# On success, freezes the resolved versions into
#   envs/quant-requirements.lock.txt
# which is the reproducibility artifact — the .in file records intent only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-quant"

echo "[bootstrap-quant] venv: $VENV"
uv venv --allow-existing "$VENV"

# If a lockfile exists, reproduce it exactly; otherwise resolve from .in
if [[ -s "$ROOT/envs/quant-requirements.lock.txt" ]]; then
    echo "[bootstrap-quant] Installing from lockfile (reproducible)"
    uv pip install --python "$VENV/bin/python" -r "$ROOT/envs/quant-requirements.lock.txt"
else
    echo "[bootstrap-quant] No lockfile — resolving from quant-requirements.in"
    uv pip install --python "$VENV/bin/python" -r "$ROOT/envs/quant-requirements.in"
    uv pip freeze --python "$VENV/bin/python" > "$ROOT/envs/quant-requirements.lock.txt"
    echo "[bootstrap-quant] Lockfile written: envs/quant-requirements.lock.txt"
fi

"$VENV/bin/python" - <<'EOF'
import torch, transformers, datasets, safetensors
print(f"[bootstrap-quant] OK  torch={torch.__version__}  cuda={torch.version.cuda}  "
      f"cuda_available={torch.cuda.is_available()}  "
      f"transformers={transformers.__version__}")
EOF
echo "[bootstrap-quant] Done. Activate with: source $VENV/bin/activate"
