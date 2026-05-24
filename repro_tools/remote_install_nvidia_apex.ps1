$ErrorActionPreference = "Stop"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
$log = Join-Path $repo "repro_logs\install_nvidia_apex.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
wsl.exe -d ADAPT-Ubuntu -- bash -lc "set -e; source /opt/conda/etc/profile.d/conda.sh; conda activate adapt; cd /mnt/e/sbw/ADAPT_repro; python -m pip uninstall -y apex || true; rm -rf downloads/nvidia_apex; mkdir -p downloads; git clone --depth 1 https://github.com/NVIDIA/apex.git downloads/nvidia_apex; cd downloads/nvidia_apex; python -m pip install -v --no-build-isolation . 2>&1 | tee /mnt/e/sbw/ADAPT_repro/ADAPT/repro_logs/install_nvidia_apex.log; python - <<'PY'
from apex import amp
print('nvidia_apex_amp_ok')
PY"
Write-Host "---APEX INSTALL TAIL---"
Get-Content $log -Tail 80
