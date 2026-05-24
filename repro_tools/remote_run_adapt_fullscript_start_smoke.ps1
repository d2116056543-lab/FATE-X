$ErrorActionPreference = "Stop"
wsl -d ADAPT-Ubuntu -- bash -lc "cd /mnt/e/sbw/ADAPT_repro/ADAPT && SMOKE_TIMEOUT_SECONDS=240 bash scripts/ADAPT_single_gpu_fullscript_start_smoke.sh"
