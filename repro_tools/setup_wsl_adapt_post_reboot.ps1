param(
  [string]$DistroName = "ADAPT-Ubuntu",
  [string]$InstallRoot = "F:\sbw_adapt_assets\wsl\ADAPT-Ubuntu",
  [string]$Rootfs = "F:\sbw_adapt_assets\wsl\ubuntu-jammy-wsl-amd64-wsl.rootfs.tar.gz",
  [string]$RepoWin = "E:\sbw\ADAPT_repro\ADAPT",
  [switch]$SkipLinuxEnvInstall
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ""
  Write-Host "==== $Message ===="
}

Write-Step "Checking pending reboot"
$cbsPending = Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"
$wuPending = Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"
Write-Host "CBS pending reboot: $cbsPending"
Write-Host "Windows Update pending reboot: $wuPending"
if ($cbsPending -or $wuPending) {
  throw "Windows still has a pending reboot. Reboot first, then rerun this script."
}

Write-Step "Checking Windows WSL features"
$features = @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")
foreach ($feature in $features) {
  $state = (Get-WindowsOptionalFeature -Online -FeatureName $feature).State
  Write-Host "$feature = $state"
  if ($state -ne "Enabled") {
    throw "$feature is not Enabled. Run DISM enable-feature for this feature, reboot, then rerun this script."
  }
}

Write-Step "Checking WSL command"
wsl.exe --status
wsl.exe --set-default-version 2

if (!(Test-Path $Rootfs)) {
  throw "Ubuntu WSL rootfs not found: $Rootfs"
}

if (!(Test-Path $InstallRoot)) {
  New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
}

Write-Step "Importing or reusing WSL distro $DistroName"
$existing = (wsl.exe -l -q) | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ }
if ($existing -contains $DistroName) {
  Write-Host "Distro already exists: $DistroName"
} else {
  wsl.exe --import $DistroName $InstallRoot $Rootfs --version 2
}

Write-Step "Verifying WSL distro"
wsl.exe -d $DistroName -- bash -lc "cat /etc/os-release && uname -a && nvidia-smi || true && ls -ld /mnt/e/sbw/ADAPT_repro/ADAPT /mnt/f/sbw_adapt_assets || true"

Write-Step "Checking Linux setup script"
$setupScript = Join-Path $RepoWin "repro_tools\setup_adapt_linux_env.sh"
if (!(Test-Path $setupScript)) {
  throw "Missing Linux setup script: $setupScript"
}

if ($SkipLinuxEnvInstall) {
  Write-Host "SkipLinuxEnvInstall was set. WSL import is complete; Linux env install was not run."
  exit 0
}

Write-Step "Running Linux ADAPT environment setup"
wsl.exe -d $DistroName -- bash /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools/setup_adapt_linux_env.sh

Write-Step "Running Linux ADAPT environment verification"
wsl.exe -d $DistroName -- bash /mnt/e/sbw/ADAPT_repro/ADAPT/repro_tools/verify_adapt_linux_env.sh

Write-Step "Done"
