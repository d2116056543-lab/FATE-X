$ErrorActionPreference = "Stop"
wsl -d ADAPT-Ubuntu -- bash -lc "cd /mnt/e/sbw/ADAPT_repro/ADAPT && SMOKE_TIMEOUT_SECONDS=120 bash scripts/ADAPT_single_gpu_eval_smoke_timeout.sh"
