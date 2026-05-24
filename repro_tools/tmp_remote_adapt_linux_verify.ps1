$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
Get-Item @(
  ".\repro_tools\setup_wsl_adapt_post_reboot.ps1",
  ".\repro_tools\setup_adapt_linux_env.sh",
  ".\repro_tools\verify_adapt_linux_env.sh",
  ".\LINUX_WSL_SETUP_STATUS.md"
) | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize

$ps = Get-Content -Raw ".\repro_tools\setup_wsl_adapt_post_reboot.ps1"
[scriptblock]::Create($ps) | Out-Null
Write-Host "Remote PowerShell parse OK"

Write-Host "Rootfs exists:" (Test-Path "F:\sbw_adapt_assets\wsl\ubuntu-jammy-wsl-amd64-wsl.rootfs.tar.gz")
Write-Host "Repo exists in E:" (Test-Path "E:\sbw\ADAPT_repro\ADAPT")
Write-Host "Asset root exists in F:" (Test-Path "F:\sbw_adapt_assets")
