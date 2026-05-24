$ErrorActionPreference = "Stop"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
$logDir = Join-Path $repo "repro_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "single_gpu_smoke.log"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "cd /mnt/e/sbw/ADAPT_repro/ADAPT && OUTPUT_DIR=./output/repro_single_gpu/smoke_multitask bash scripts/ADAPT_single_gpu_smoke.sh 2>&1 | tee repro_logs/single_gpu_smoke.log"
Write-Host "---SMOKE LOG TAIL---"
Get-Content $log -Tail 80
