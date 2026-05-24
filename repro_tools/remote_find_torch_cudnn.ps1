$ErrorActionPreference = "Stop"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "chmod +x /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools/find_torch_cudnn.sh && /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools/find_torch_cudnn.sh"
