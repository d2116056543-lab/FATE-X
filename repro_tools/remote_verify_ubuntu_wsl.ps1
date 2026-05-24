$ErrorActionPreference = "Continue"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
Write-Host "---WSL LIST---"
& wsl.exe -l -v
Write-Host "---WSL BASIC---"
& wsl.exe -d Ubuntu-22.04 -- bash -lc 'echo WSL_OK; whoami; pwd; cat /etc/os-release | head -5'
Write-Host "---WSL PATHS---"
& wsl.exe -d Ubuntu-22.04 -- bash -lc 'ls -ld /mnt/e/sbw/ADAPT_repro/ADAPT /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools; ls -ld /mnt/f/sbw_adapt_assets 2>&1 || true'
Write-Host "---WSL GPU---"
& wsl.exe -d Ubuntu-22.04 -- bash -lc 'nvidia-smi | head -8'
