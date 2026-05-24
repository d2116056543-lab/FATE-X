$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
Write-Host "---RUN LINUX SETUP---"
wsl.exe -d ADAPT-Ubuntu -- bash /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools/setup_adapt_linux_env.sh
Write-Host "setup_exit=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
