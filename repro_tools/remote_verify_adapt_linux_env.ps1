$ErrorActionPreference = "Stop"

wsl.exe -d ADAPT-Ubuntu -- bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate adapt && cd /mnt/e/sbw/ADAPT_repro/ADAPT && python /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools/verify_adapt_linux_env.py"
