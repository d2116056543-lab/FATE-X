$ErrorActionPreference = "Continue"
Get-CimInstance Win32_Process |
  Where-Object { $_.ProcessId -in @(14984,15092,3212,9100) } |
  Select-Object ProcessId,Name,CommandLine |
  Format-List
