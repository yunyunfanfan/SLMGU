#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-slmgu}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
CUDA_TAG=${CUDA_TAG:-cu121}

printf "[1/4] Checking Python/Conda...\n"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
  fi
  conda activate "$ENV_NAME"
else
  echo "Conda not found. Using current Python environment: $(python --version)"
fi

printf "[2/4] Installing PyTorch/PyG dependencies...\n"
python -m pip install --upgrade pip
if python - <<'PYCHK' >/dev/null 2>&1
import torch, torch_geometric
PYCHK
then
  echo "PyTorch and PyG already import successfully; installing remaining requirements only."
  python -m pip install -r requirements.txt
else
  python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
  python -m pip install torch-geometric
  python -m pip install -r requirements.txt
fi

printf "[3/4] Checking CUDA...\n"
python - <<'PYCUDA'
import torch
print('Python/PyTorch OK')
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PYCUDA

printf "[4/4] Creating local artifact directories...\n"
mkdir -p datasets checkpoints outputs/logs outputs/results outputs/checkpoints

echo "Installation finished. Prepare datasets/ and checkpoints/ before reproduction."
