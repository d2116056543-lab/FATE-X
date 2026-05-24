$ErrorActionPreference = "Continue"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
Set-Location $repo

Write-Host "---PWD---"
Get-Location

Write-Host "---KEY PATHS---"
$paths = @(
  "ADAPT_PREPROCESSED_DATASET",
  "ADAPT_PREPROCESSED_DATASET\datasets",
  "ADAPT_PREPROCESSED_DATASET\datasets_part",
  "datasets",
  "datasets\BDDX",
  "datasets\BDDX_des",
  "datasets\BDDX_exp",
  "datasets_part",
  "models",
  "checkpoints",
  "src\evalcap"
)
foreach ($p in $paths) {
  if (Test-Path $p) {
    $item = Get-Item $p -Force
    Write-Host ("OK {0} mode={1} target={2}" -f $p, $item.Mode, $item.Target)
  } else {
    Write-Host ("MISSING {0}" -f $p)
  }
}

Write-Host "---DATASET README---"
if (Test-Path "ADAPT_PREPROCESSED_DATASET\Readme.txt") {
  Get-Content "ADAPT_PREPROCESSED_DATASET\Readme.txt" -Raw
}

Write-Host "---YAML COUNTS---"
$yamls = @(
  "datasets\BDDX\training_32frames.yaml",
  "datasets\BDDX\validation_32frames.yaml",
  "datasets\BDDX\testing_32frames.yaml",
  "datasets_part\BDDX\training_32frames.yaml",
  "datasets_part\BDDX\validation_32frames.yaml",
  "datasets_part\BDDX\testing_32frames.yaml",
  "datasets\BDDX_des\testing_32frames.yaml",
  "datasets\BDDX_exp\testing_32frames.yaml"
)
foreach ($y in $yamls) {
  if (Test-Path $y) {
    Write-Host ("--- {0} ---" -f $y)
    Get-Content $y
  } else {
    Write-Host ("MISSING {0}" -f $y)
  }
}

Write-Host "---SCRIPT HEADS---"
$scripts = @(
  "scripts\BDDX_multitask.sh",
  "scripts\ADAPT_single_gpu_multitask.sh",
  "scripts\ADAPT_single_gpu_eval.sh",
  "scripts\ADAPT_single_gpu_smoke.sh"
)
foreach ($s in $scripts) {
  if (Test-Path $s) {
    Write-Host ("--- {0} ---" -f $s)
    Get-Content $s -TotalCount 80
  } else {
    Write-Host ("MISSING {0}" -f $s)
  }
}

Write-Host "---ACTIVE ADAPT PROCESSES---"
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -and ($_.CommandLine -match "ADAPT_single_gpu|run_adapt.py|torch.distributed.launch") } |
  Select-Object ProcessId,Name,CommandLine |
  Format-List

Write-Host "---SMOKE LOG TAIL---"
if (Test-Path "repro_logs\single_gpu_smoke.log") {
  Get-Content "repro_logs\single_gpu_smoke.log" -Tail 80
}

Write-Host "---VALIDATION JSON HEAD---"
if (Test-Path "repro_logs\adapt_preprocessed_dataset_validation.json") {
  Get-Content "repro_logs\adapt_preprocessed_dataset_validation.json" -TotalCount 120
}
