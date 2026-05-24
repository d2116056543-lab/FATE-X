$ErrorActionPreference = "Continue"
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "ADAPT_single_gpu_smoke|run_adapt.py|torch.distributed.launch" } |
  Select-Object ProcessId,Name,CommandLine |
  Format-List
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "ADAPT_single_gpu_smoke|run_adapt.py|torch.distributed.launch" } |
  ForEach-Object {
    Write-Host "stopping $($_.ProcessId) $($_.Name)"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }
