$ErrorActionPreference = "Continue"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
New-Item -ItemType Directory -Force -Path ".\repro_logs\linux_setup" | Out-Null

Write-Host "---HOST---"
hostname
whoami
Write-Host "---WSL VERSION---"
& wsl.exe --version
Write-Host "version_exit=$LASTEXITCODE"
Write-Host "---WSL LIST BEFORE---"
& wsl.exe -l -v
Write-Host "list_before_exit=$LASTEXITCODE"

$distro = "ADAPT-Ubuntu"
$installRoot = "F:\sbw_adapt_assets\wsl\ADAPT-Ubuntu"
$rootfs = "F:\sbw_adapt_assets\wsl\ubuntu-jammy-wsl-amd64-wsl.rootfs.tar.gz"

Write-Host "---PATH CHECK---"
Write-Host "rootfs_exists=$(Test-Path $rootfs)"
Write-Host "install_root_exists=$(Test-Path $installRoot)"
if (!(Test-Path $rootfs)) {
  throw "Rootfs missing: $rootfs"
}

$existing = (& wsl.exe -l -q) | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ }
if ($existing -contains $distro) {
  Write-Host "$distro already exists"
} else {
  if (Test-Path $installRoot) {
    $children = Get-ChildItem $installRoot -Force -ErrorAction SilentlyContinue
    if ($children.Count -eq 0) {
      Remove-Item $installRoot -Force
    }
  }
  New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
  Write-Host "---IMPORT---"
  & wsl.exe --import $distro $installRoot $rootfs --version 2 *> ".\repro_logs\linux_setup\wsl_import_adapt_ubuntu.log"
  Write-Host "import_exit=$LASTEXITCODE"
  Get-Content ".\repro_logs\linux_setup\wsl_import_adapt_ubuntu.log" -Raw -ErrorAction SilentlyContinue
}

Write-Host "---WSL LIST AFTER---"
& wsl.exe -l -v
Write-Host "list_after_exit=$LASTEXITCODE"
Write-Host "---RUN CHECK---"
& wsl.exe -d $distro -- bash -lc 'echo WSL_OK; whoami; cat /etc/os-release | head -5; ls -ld /mnt/e/sbw/ADAPT_repro/ADAPT /mnt/f/sbw_adapt_assets; nvidia-smi | head -8'
Write-Host "run_check_exit=$LASTEXITCODE"
