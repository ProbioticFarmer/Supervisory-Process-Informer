#!/usr/bin/env bash
set -euo pipefail

# Raise the macOS Metal wired memory limit to ~115 GB (117760 MB) on 128 GB M4 Max:
# sudo sysctl iogpu.wired_limit_mb=117760

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/3] Creating Python 3.12 virtual environment (.venv)..."
if command -v python3.12 &>/dev/null; then
    python3.12 -m venv .venv
elif command -v uv &>/dev/null; then
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRIPT_DIR/.uv-python}"
    uv venv --seed --python 3.12 .venv
elif [ -x /opt/homebrew/bin/python3 ]; then
    /opt/homebrew/bin/python3 -m venv .venv
else
    python3 -m venv .venv
fi

echo "[2/3] Upgrading pip and installing MLX interpretability dependencies..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install \
    mlx \
    mlx-lm \
    huggingface_hub \
    safetensors \
    gguf \
    numpy \
    matplotlib \
    fastapi \
    uvicorn \
    pydantic \
    httpx

echo "[3/3] Freezing requirements.txt..."
.venv/bin/pip freeze > requirements.txt

echo "Environment setup complete. Activate with: source .venv/bin/activate"
