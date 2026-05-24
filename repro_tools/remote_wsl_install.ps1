$ErrorActionPreference = "Continue"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
New-Item -ItemType Directory -Force -Path ".\repro_logs\linux_setup" | Out-Null
Write-Host "---HOST---"
hostname
whoami
Write-Host "---INSTALL---"
& wsl.exe --install *> ".\repro_logs\linux_setup\wsl_install.log"
Write-Host "install_exit=$LASTEXITCODE"
Get-Content ".\repro_logs\linux_setup\wsl_install.log" -Raw -ErrorAction SilentlyContinue
Write-Host "---PENDING---"
Write-Host ("CBS=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"))
Write-Host ("WU=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"))
