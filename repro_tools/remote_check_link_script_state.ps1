$ErrorActionPreference = "Continue"
Write-Host "---PROCESSES---"
Get-Process | Where-Object { $_.ProcessName -match 'cmd|powershell|robocopy' } | Select-Object Id,ProcessName,CPU,Path | Format-Table -AutoSize
Write-Host "---LINKS---"
cmd /c dir /AL E:\sbw\ADAPT_repro\ADAPT
cmd /c dir /AL E:\sbw\ADAPT_repro\ADAPT\datasets
Write-Host "---EXISTENCE---"
foreach ($p in @(
  "E:\sbw\ADAPT_repro\ADAPT\datasets\BDDX\training_32frames.yaml",
  "E:\sbw\ADAPT_repro\ADAPT\datasets_part\BDDX\training_32frames.yaml",
  "E:\sbw\ADAPT_repro\ADAPT\ADAPT_PREPROCESSED_DATASET\datasets\BDDX\training_32frames.yaml"
)) {
  if (Test-Path $p) { "OK $p" } else { "MISSING $p" }
}
