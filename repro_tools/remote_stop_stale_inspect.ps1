$ErrorActionPreference = "Continue"
$candidates = Get-Process | Where-Object {
  ($_.ProcessName -match 'findstr') -or
  ($_.ProcessName -eq 'powershell' -and $_.CPU -gt 60)
}
$candidates | Select-Object Id,ProcessName,CPU | Format-Table -AutoSize
foreach ($p in $candidates) {
  try {
    Stop-Process -Id $p.Id -Force
    Write-Host "stopped $($p.Id) $($p.ProcessName)"
  } catch {
    Write-Host "failed to stop $($p.Id): $($_.Exception.Message)"
  }
}
