$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"

$distro = "ADAPT-Ubuntu"

Write-Host "--- INSTALL TORCH FROM LOCAL WHEELS ---"
wsl.exe -d $distro -- bash -lc @'
set -euxo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

python -m pip install -U pip
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.timeout 120
python -m pip install --retries 20 --timeout 120 -U typing-extensions numpy pillow requests packaging filelock sympy networkx jinja2

cd /mnt/e/sbw/ADAPT_repro/downloads/torch_cu117
ls -lh
python -m pip install --retries 20 --timeout 120 \
  ./torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl \
  ./torchvision-0.14.1+cu117-cp38-cp38-linux_x86_64.whl \
  ./torchaudio-0.13.1+cu117-cp38-cp38-linux_x86_64.whl

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
'@
