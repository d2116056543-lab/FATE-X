$ErrorActionPreference = "Stop"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
$log = Join-Path $repo "repro_logs\install_legacy_apex_amp.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
wsl.exe -d ADAPT-Ubuntu -- bash -lc "set -e; source /opt/conda/etc/profile.d/conda.sh; conda activate adapt; cd /mnt/e/sbw/ADAPT_repro; python -m pip uninstall -y apex || true; rm -rf downloads/nvidia_apex_legacy; mkdir -p downloads; git clone --depth 200 https://github.com/NVIDIA/apex.git downloads/nvidia_apex_legacy; cd downloads/nvidia_apex_legacy; git checkout e13873de || git checkout b9d758c0; python -m pip install -v --no-build-isolation . 2>&1 | tee /mnt/e/sbw/ADAPT_repro/ADAPT/repro_logs/install_legacy_apex_amp.log; cd /mnt/e/sbw/ADAPT_repro/ADAPT; python - <<'PY'
from apex import amp
print('legacy_apex_amp_ok')
PY"
Write-Host "---LEGACY APEX INSTALL TAIL---"
Get-Content $log -Tail 80
