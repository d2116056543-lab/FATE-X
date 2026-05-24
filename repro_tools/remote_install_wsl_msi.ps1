$ErrorActionPreference = "Continue"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
New-Item -ItemType Directory -Force -Path ".\repro_logs\linux_setup" | Out-Null
New-Item -ItemType Directory -Force -Path "E:\sbw\ADAPT_repro\downloads" | Out-Null

$url = "https://github.com/microsoft/WSL/releases/download/2.7.3/wsl.2.7.3.0.x64.msi"
$msi = "E:\sbw\ADAPT_repro\downloads\wsl.2.7.3.0.x64.msi"

Write-Host "---HOST---"
hostname
whoami

Write-Host "---DOWNLOAD MSI---"
curl.exe -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -o $msi $url
Write-Host "curl_exit=$LASTEXITCODE"
Get-Item $msi -ErrorAction SilentlyContinue | Select-Object FullName,Length,LastWriteTime | Format-List

if (!(Test-Path $msi) -or ((Get-Item $msi).Length -lt 200MB)) {
  throw "WSL MSI download incomplete: $msi"
}

Write-Host "---INSTALL MSI---"
$p = Start-Process msiexec.exe -ArgumentList @("/i", $msi, "/quiet", "/norestart") -Wait -PassThru
Write-Host "msiexec_exit=$($p.ExitCode)"

Write-Host "---WSL VERSION---"
& wsl.exe --version
Write-Host "version_exit=$LASTEXITCODE"

Write-Host "---PENDING---"
Write-Host ("CBS=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"))
Write-Host ("WU=" + (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"))
