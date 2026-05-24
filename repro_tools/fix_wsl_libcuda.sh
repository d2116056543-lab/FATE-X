#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

echo "---conda---"
echo "CONDA_PREFIX=${CONDA_PREFIX}"

echo "---wsl cuda libs---"
ls -l /usr/lib/wsl/lib/libcuda* || true

echo "---conda cudnn libs---"
ls -l "${CONDA_PREFIX}"/lib/libcudnn* | head || true

if [ -f /usr/lib/wsl/lib/libcuda.so.1 ] && [ ! -e /usr/lib/wsl/lib/libcuda.so ]; then
  ln -s /usr/lib/wsl/lib/libcuda.so.1 /usr/lib/wsl/lib/libcuda.so
fi

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/adapt_cuda_paths.sh" <<'SH'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
SH

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
python - <<'PY'
import ctypes
import torch

print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
ctypes.CDLL("libcuda.so")
ctypes.CDLL("libcudnn_cnn_infer.so.8")
print("libcuda_and_cudnn_load_ok")
PY
