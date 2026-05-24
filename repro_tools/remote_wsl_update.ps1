$ErrorActionPreference = "Continue"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
New-Item -ItemType Directory -Force -Path ".\repro_logs\linux_setup" | Out-Null

Write-Host "---HOST---"
hostname
whoami
Write-Host "---BEFORE---"
& wsl.exe --version *> ".\repro_logs\linux_setup\wsl_version_before_update.raw.txt"
Write-Host "before_version_exit=$LASTEXITCODE"
Get-Content ".\repro_logs\linux_setup\wsl_version_before_update.raw.txt" -Raw -ErrorAction SilentlyContinue

Write-Host "---UPDATE---"
& wsl.exe --update *> ".\repro_logs\linux_setup\wsl_update.log"
Write-Host "update_exit=$LASTEXITCODE"
Get-Content ".\repro_logs\linux_setup\wsl_update.log" -Raw -ErrorAction SilentlyContinue

Write-Host "---AFTER---"
& wsl.exe --version *> ".\repro_logs\linux_setup\wsl_version_after_update.raw.txt"
Write-Host "after_version_exit=$LASTEXITCODE"
Get-Content ".\repro_logs\linux_setup\wsl_version_after_update.raw.txt" -Raw -ErrorAction SilentlyContinue

Write-Host "---PENDING---"
Write-Host ("CBS=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"))
Write-Host ("WU=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"))
