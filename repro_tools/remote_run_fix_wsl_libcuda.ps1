$ErrorActionPreference = "Stop"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "chmod +x /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools/fix_wsl_libcuda.sh && /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools/fix_wsl_libcuda.sh"
