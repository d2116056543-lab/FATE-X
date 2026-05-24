$ErrorActionPreference = "Continue"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
Write-Host "---PWD---"
Get-Location | Format-List
Write-Host "---WSL STATUS---"
wsl.exe --status 2>&1
Write-Host "---WSL LIST---"
wsl.exe -l -v 2>&1
Write-Host "---FEATURES---"
foreach ($feature in @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")) {
  Get-WindowsOptionalFeature -Online -FeatureName $feature |
    Select-Object FeatureName, State |
    Format-List
}
Write-Host "---REBOOT PENDING---"
Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"
Write-Host "---ROOTFS---"
Get-Item "F:\sbw_adapt_assets\wsl\ubuntu-jammy-wsl-amd64-wsl.rootfs.tar.gz" -ErrorAction SilentlyContinue |
  Select-Object FullName, Length, LastWriteTime |
  Format-List
Write-Host "---REPO FILES---"
foreach ($path in @("requirements.txt", "src", "repro_tools", "datasets\BDDX", "checkpoints", "models")) {
  Write-Host "$path =" (Test-Path $path)
}
