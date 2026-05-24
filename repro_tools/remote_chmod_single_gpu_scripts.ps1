$ErrorActionPreference = "Stop"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "cd /mnt/e/sbw/ADAPT_repro/ADAPT && chmod +x scripts/ADAPT_single_gpu_*.sh && ls -l scripts/ADAPT_single_gpu_*.sh"
