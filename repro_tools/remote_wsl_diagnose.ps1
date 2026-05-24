$ErrorActionPreference = "Continue"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
New-Item -ItemType Directory -Force -Path ".\repro_logs\linux_setup" | Out-Null

Write-Host "---IDENTITY---"
hostname
whoami
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Write-Host "is_admin=$isAdmin"

Write-Host "---WSL COMMAND---"
where.exe wsl
Get-Command wsl.exe | Format-List Source,Version

Write-Host "---WSL VERSION---"
& wsl.exe --version *> ".\repro_logs\linux_setup\wsl_version.raw.txt"
Write-Host "version_exit=$LASTEXITCODE"
Get-Content ".\repro_logs\linux_setup\wsl_version.raw.txt" -Raw -ErrorAction SilentlyContinue

Write-Host "---WSL HELP---"
& wsl.exe --help *> ".\repro_logs\linux_setup\wsl_help.raw.txt"
Write-Host "help_exit=$LASTEXITCODE"
Get-Content ".\repro_logs\linux_setup\wsl_help.raw.txt" -TotalCount 80 -ErrorAction SilentlyContinue

Write-Host "---APPX WSL---"
Get-AppxPackage -Name "*WindowsSubsystemForLinux*" | Select-Object Name,PackageFullName,Version,InstallLocation | Format-List

Write-Host "---FEATURES---"
foreach ($feature in @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform", "HypervisorPlatform")) {
  try {
    Get-WindowsOptionalFeature -Online -FeatureName $feature | Select-Object FeatureName,State | Format-List
  } catch {
    Write-Host "feature_check_failed $feature $($_.Exception.Message)"
  }
}
