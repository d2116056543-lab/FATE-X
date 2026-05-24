$ErrorActionPreference = "Continue"
Write-Host "---CURL PROCESSES---"
Get-Process curl -ErrorAction SilentlyContinue | Select-Object Id, CPU, StartTime, Path | Format-Table -AutoSize
Write-Host "---MSI FILE---"
Get-Item "E:\sbw\ADAPT_repro\downloads\wsl.2.7.3.0.x64.msi" -ErrorAction SilentlyContinue |
  Select-Object FullName, Length, LastWriteTime |
  Format-List
Write-Host "---WSL VERSION---"
wsl.exe --version
Write-Host "wsl_exit=$LASTEXITCODE"
