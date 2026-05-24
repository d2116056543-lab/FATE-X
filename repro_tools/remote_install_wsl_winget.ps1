$ErrorActionPreference = "Continue"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
New-Item -ItemType Directory -Force -Path ".\repro_logs\linux_setup" | Out-Null

Write-Host "---HOST---"
hostname
whoami
Write-Host "---WINGET---"
where.exe winget
& winget -v

Write-Host "---WSL PACKAGE BEFORE---"
& winget list --id Microsoft.WSL -e --accept-source-agreements *> ".\repro_logs\linux_setup\winget_wsl_list_before.log"
Write-Host "list_before_exit=$LASTEXITCODE"
Get-Content ".\repro_logs\linux_setup\winget_wsl_list_before.log" -Raw -ErrorAction SilentlyContinue

Write-Host "---INSTALL MICROSOFT.WSL---"
& winget install --id Microsoft.WSL -e --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity *> ".\repro_logs\linux_setup\winget_install_microsoft_wsl.log"
Write-Host "install_exit=$LASTEXITCODE"
Get-Content ".\repro_logs\linux_setup\winget_install_microsoft_wsl.log" -Raw -ErrorAction SilentlyContinue

Write-Host "---WSL VERSION AFTER---"
& wsl.exe --version *> ".\repro_logs\linux_setup\wsl_version_after_winget.log"
Write-Host "version_exit=$LASTEXITCODE"
Get-Content ".\repro_logs\linux_setup\wsl_version_after_winget.log" -Raw -ErrorAction SilentlyContinue

Write-Host "---PENDING---"
Write-Host ("CBS=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"))
Write-Host ("WU=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"))
