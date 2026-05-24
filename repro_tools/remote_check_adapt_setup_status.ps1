$ErrorActionPreference = "Continue"

Write-Host "---PROCESSES---"
Get-Process |
  Where-Object { $_.ProcessName -match 'aria2|curl|wget|pip|conda|BaiduNetdisk|python|wsl' } |
  Select-Object Id, ProcessName, CPU, WorkingSet |
  Format-Table -AutoSize

Write-Host "---SETUP LOG TAIL---"
$log = "E:\sbw\ADAPT_repro\ADAPT\repro_logs\linux_setup\setup_adapt_linux_env.log"
if (Test-Path $log) {
  Get-Content $log -Tail 40
} else {
  Write-Host "missing log: $log"
}

Write-Host "---WSL DISTROS---"
wsl.exe -l -v
