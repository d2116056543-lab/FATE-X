$ErrorActionPreference = "Continue"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
$runName = "adapt_full_b4a16_20260522_0036"
$logPath = Join-Path $repo "repro_logs\$runName.log"
$outDir = Join-Path $repo "output\repro_single_gpu\$runName"

Write-Host "---RUN---"
Write-Host $runName
Write-Host "---PROCESS---"
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -and ($_.CommandLine -match $runName -or $_.CommandLine -match "run_adapt.py") } |
  Select-Object ProcessId,Name,CommandLine |
  Format-List

Write-Host "---GPU---"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

Write-Host "---LOG TAIL---"
if (Test-Path $logPath) {
  Get-Content $logPath -Tail 80
} else {
  Write-Host "log not created yet: $logPath"
}

Write-Host "---CHECKPOINTS---"
if (Test-Path $outDir) {
  Get-ChildItem $outDir -Force |
    Where-Object { $_.Name -match "checkpoint|restore|log" } |
    Select-Object Name,Length,LastWriteTime |
    Format-Table -AutoSize
  if (Test-Path (Join-Path $outDir "checkpoint_latest\repro_checkpoint_meta.json")) {
    Write-Host "---LATEST META---"
    Get-Content (Join-Path $outDir "checkpoint_latest\repro_checkpoint_meta.json") -Raw
  }
  if (Test-Path (Join-Path $outDir "checkpoint_best\repro_checkpoint_meta.json")) {
    Write-Host "---BEST META---"
    Get-Content (Join-Path $outDir "checkpoint_best\repro_checkpoint_meta.json") -Raw
  }
} else {
  Write-Host "output not created yet: $outDir"
}
