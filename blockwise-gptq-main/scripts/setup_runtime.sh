#!/usr/bin/env bash
# scripts/setup_runtime.sh
#
# Sets up the Python environment for Block-wise GPTQ · GPT-OSS 20B · NVFP4.
# Installs PyTorch (CUDA-aware) + project dependencies.
#
# Usage (run once from project root):
#   bash scripts/setup_runtime.sh
#
# Environment overrides:
#   VENV          — path to venv        (default: <repo_root>/.venv)
#   CUDA_VERSION  — override CUDA ver   (e.g. "12.8")
#   TORCH_VERSION — PyTorch version     (default: "2.7.0")
#
# Requirements:
#   torch >= 2.1.0 for float8_e4m3fn (NVFP4 microscaling scales)
#   CUDA  >= 12.0  for float8 support

set -euo pipefail

# ── Resolve project root ─────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-${ROOT}/.venv}"
REQ="${ROOT}/requirements.txt"

echo "======================================"
echo " Block-wise GPTQ · GPT-OSS 20B · NVFP4"
echo " Root : $ROOT"
echo " Venv : $VENV"
echo "======================================"

# ── [1/6] NVIDIA environment check ───────────────────────────────────────────
echo ""
echo "[1/6] Checking NVIDIA environment..."

if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed — driver not installed or unresponsive."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Detect CUDA toolkit version (needed to pick the right PyTorch wheel)
if [[ -n "${CUDA_VERSION:-}" ]]; then
    CUDA_VER="$CUDA_VERSION"
elif command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
else
    echo "WARNING: nvcc not found; defaulting to CUDA 12.8"
    CUDA_VER="12.8"
fi
echo "Detected CUDA version: $CUDA_VER"

# float8_e4m3fn requires CUDA >= 12.0; warn if older
CUDA_MAJOR="${CUDA_VER%%.*}"
if [[ "$CUDA_MAJOR" -lt 12 ]]; then
    echo "WARNING: CUDA < 12.0 detected. float8_e4m3fn (required by NVFP4Quantizer)"
    echo "         will not be available. Stage 1 Test 2 will be skipped."
fi

# Map CUDA version → PyTorch wheel index
# DGX Spark GB10 ships with CUDA 12.8+
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
case "$CUDA_VER" in
    12.8|12.9|13.*) TORCH_IDX="https://download.pytorch.org/whl/cu128" ;;
    12.6|12.7)      TORCH_IDX="https://download.pytorch.org/whl/cu126" ;;
    12.4|12.5)      TORCH_IDX="https://download.pytorch.org/whl/cu124" ;;
    *)
        echo "WARNING: unknown CUDA version '$CUDA_VER'; defaulting to cu128"
        TORCH_IDX="https://download.pytorch.org/whl/cu128" ;;
esac
echo "PyTorch ${TORCH_VERSION} wheel index: $TORCH_IDX"
echo "✔ GPU environment OK"

# ── [2/6] System dependencies ─────────────────────────────────────────────────
echo ""
echo "[2/6] Installing system dependencies..."

SUDO=""
if [[ "$EUID" -ne 0 ]]; then
    SUDO="sudo"
fi

PY_MINOR=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

$SUDO apt-get update -qq && $SUDO apt-get install -y \
    git \
    git-lfs \
    cmake \
    build-essential \
    ninja-build \
    python3 \
    python3-dev \
    "python${PY_MINOR}-dev"

git lfs install --skip-repo
echo "✔ System dependencies installed"

# ── [3/6] Validate requirements file ─────────────────────────────────────────
echo ""
echo "[3/6] Checking requirements file..."
if [[ ! -f "$REQ" ]]; then
    echo "ERROR: requirements.txt not found at $REQ"
    exit 1
fi
echo "✔ Found $REQ"

# ── [4/6] Create / reuse virtual environment ──────────────────────────────────
echo ""
echo "[4/6] Virtual environment..."
if [[ -d "$VENV" ]]; then
    echo "Reusing existing venv at $VENV"
else
    python3 -m venv "$VENV"
    echo "Created venv at $VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"
echo "Active Python: $(which python) — $(python --version)"
pip install --quiet --upgrade pip

# ── [5/6] PyTorch ─────────────────────────────────────────────────────────────
echo ""
echo "[5/6] Installing PyTorch ${TORCH_VERSION} for CUDA ${CUDA_VER}..."
pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_IDX"

# Verify float8 support is present (required by NVFP4Quantizer)
python - <<'EOF'
import torch, sys
if not hasattr(torch, "float8_e4m3fn"):
    print("WARNING: torch.float8_e4m3fn not available — "
          "Stage 1 Test 2 (FP8 scale quantization) will be skipped.")
else:
    print(f"✔ float8_e4m3fn available (torch {torch.__version__})")
EOF
echo "✔ PyTorch installed"

# ── [6/6] Project requirements + environment variables ───────────────────────
echo ""
echo "[6/6] Installing project requirements..."
pip install -r "$REQ"
echo "✔ Project requirements installed"

ACTIVATE="$VENV/bin/activate"

add_env_var() {
    local var="$1"
    if ! grep -q "export ${var}" "$ACTIVATE"; then
        echo "export ${var}" >> "$ACTIVATE"
        echo "  Added: $var"
    else
        echo "  Already set: $var"
    fi
}

# Fast HuggingFace downloads (requires hf_transfer in requirements.txt)
add_env_var "HF_HUB_ENABLE_HF_TRANSFER=1"

# Suppress tokenizer parallelism warnings during eval
add_env_var "TOKENIZERS_PARALLELISM=false"

echo "✔ Environment variables set"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo " Setup complete ✅"
echo ""
echo " Activate the environment:"
echo "   source $VENV/bin/activate"
echo ""
echo " Run the pipeline stages in order:"
echo "   python test/stage1_nvfp4_unit_tests.py       # ~30 s  — gate: must all pass"
echo "   python test/stage2_nvfp4_algorithm_tests.py  # ~2 min — gate: must all pass"
echo "   python test/stage3_gpt_oss_shape_tests.py    # ~30 min — includes B sweep"
echo "   python test/stage4_baseline_perplexity.py    # ~20 min — record baseline ppl"
echo "   python test/stage5_quantize_model.py         # ~60 min — produces quantized model"
echo "   python test/stage6_eval_perplexity.py        # ~20 min — Δppl vs baseline"
echo "======================================"