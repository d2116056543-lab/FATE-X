$ErrorActionPreference = "Stop"
wsl -d ADAPT-Ubuntu -- bash -lc "cd /mnt/e/sbw/ADAPT_repro/ADAPT && SMOKE_TIMEOUT_SECONDS=180 bash scripts/ADAPT_single_gpu_latest_ckpt_smoke.sh"
Write-Host "---LATEST CKPT---"
Get-ChildItem "E:\sbw\ADAPT_repro\ADAPT\output\repro_single_gpu\latest_ckpt_smoke\checkpoint_latest" -ErrorAction SilentlyContinue |
  Select-Object Name,Length,LastWriteTime |
  Format-Table -AutoSize
if (Test-Path "E:\sbw\ADAPT_repro\ADAPT\output\repro_single_gpu\latest_ckpt_smoke\checkpoint_latest\repro_checkpoint_meta.json") {
  Get-Content "E:\sbw\ADAPT_repro\ADAPT\output\repro_single_gpu\latest_ckpt_smoke\checkpoint_latest\repro_checkpoint_meta.json" -Raw
}
