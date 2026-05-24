$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"

Write-Host "---ROOTFS TAR CHECK---"
tar.exe -tzf "F:\sbw_adapt_assets\wsl\ubuntu-jammy-wsl-amd64-wsl.rootfs.tar.gz" *> "$env:TEMP\adapt_ubuntu_rootfs_tar_check.txt"
Write-Host "tar_exit=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) {
  Get-Content "$env:TEMP\adapt_ubuntu_rootfs_tar_check.txt" -Tail 20
  throw "Rootfs tar check failed"
}

Write-Host "---SCRIPT CHECK---"
$required = @(
  ".\repro_tools\setup_wsl_adapt_post_reboot.ps1",
  ".\repro_tools\setup_adapt_linux_env.sh",
  ".\repro_tools\verify_adapt_linux_env.sh",
  ".\LINUX_WSL_SETUP_STATUS.md"
)
foreach ($path in $required) {
  $item = Get-Item $path
  Write-Host "$($item.FullName) length=$($item.Length)"
}

Write-Host "---DOC HEAD---"
Get-Content ".\LINUX_WSL_SETUP_STATUS.md" -TotalCount 20
