$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
Write-Host "---INSTALL ARIA2---"
wsl.exe -d ADAPT-Ubuntu -- bash -lc 'apt-get update && apt-get install -y aria2'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "---DOWNLOAD TORCH WHEELS---"
wsl.exe -d ADAPT-Ubuntu -- bash -lc @'
set -euxo pipefail
mkdir -p /mnt/e/sbw/ADAPT_repro/downloads/torch_cu117
cd /mnt/e/sbw/ADAPT_repro/downloads/torch_cu117
aria2c -c -x 16 -s 16 -k 1M --retry-wait=5 --max-tries=0 \
  -o torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl \
  "https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl"
aria2c -c -x 8 -s 8 -k 1M --retry-wait=5 --max-tries=0 \
  -o torchvision-0.14.1+cu117-cp38-cp38-linux_x86_64.whl \
  "https://download.pytorch.org/whl/cu117/torchvision-0.14.1%2Bcu117-cp38-cp38-linux_x86_64.whl"
aria2c -c -x 8 -s 8 -k 1M --retry-wait=5 --max-tries=0 \
  -o torchaudio-0.13.1+cu117-cp38-cp38-linux_x86_64.whl \
  "https://download.pytorch.org/whl/cu117/torchaudio-0.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl"
ls -lh *.whl
'@
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "---INSTALL TORCH WHEELS---"
wsl.exe -d ADAPT-Ubuntu -- bash -lc @'
set -euxo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt
cd /mnt/e/sbw/ADAPT_repro/downloads/torch_cu117
python -m pip install --no-index \
  torch-1.13.1+cu117-cp38-cp38-linux_x86_64.whl \
  torchvision-0.14.1+cu117-cp38-cp38-linux_x86_64.whl \
  torchaudio-0.13.1+cu117-cp38-cp38-linux_x86_64.whl
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
'@
exit $LASTEXITCODE
