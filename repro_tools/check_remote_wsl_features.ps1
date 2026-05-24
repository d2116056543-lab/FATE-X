$ErrorActionPreference = "Continue"
Write-Host "---FEATURES---"
foreach ($feature in @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")) {
  try {
    Get-WindowsOptionalFeature -Online -FeatureName $feature |
      Select-Object FeatureName, State |
      Format-List
  } catch {
    Write-Host "feature_check_failed $feature $($_.Exception.Message)"
  }
}
Write-Host "---REBOOT PENDING---"
Write-Host ("CBS=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"))
Write-Host ("WU=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"))
Write-Host "---WSL STATUS---"
wsl.exe --status 2>&1
