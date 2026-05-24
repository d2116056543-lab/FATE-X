$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
wsl -d ADAPT-Ubuntu -- bash /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools/inspect_adapt_ckpt.sh
