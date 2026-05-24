$ErrorActionPreference = "Continue"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
Set-Location $repo

Write-Host "---DOCS---"
Get-ChildItem -File -Filter "ADAPT_*20260521.md" |
  Select-Object Name,Length,LastWriteTime |
  Format-Table -AutoSize

Write-Host "---SINGLE GPU SCRIPTS---"
Get-ChildItem scripts -File -Filter "ADAPT_single_gpu*.sh" |
  Select-Object Name,Length,LastWriteTime |
  Format-Table -AutoSize

Write-Host "---REPRO LOGS---"
Get-ChildItem repro_logs -File |
  Where-Object { $_.Name -match "adapt_preprocessed|single_gpu" } |
  Select-Object Name,Length,LastWriteTime |
  Sort-Object LastWriteTime |
  Format-Table -AutoSize

Write-Host "---ACTIVE ADAPT PROCESSES---"
$procs = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -and ($_.CommandLine -match "ADAPT_single_gpu|run_adapt.py|torch.distributed.launch") } |
  Select-Object ProcessId,Name,CommandLine
if ($procs) {
  $procs | Format-List
} else {
  Write-Host "none"
}

Write-Host "---FULLSCRIPT SMOKE KEY LINES---"
if (Test-Path "repro_logs\single_gpu_fullscript_start_smoke.log") {
  Select-String -Path "repro_logs\single_gpu_fullscript_start_smoke.log" -Pattern "Train with 2 images|Total training steps|input_ids =|img_feats =|car_info =|fullscript start smoke timeout" | Select-Object -Last 20
}

Write-Host "---EVAL SMOKE KEY LINES---"
if (Test-Path "repro_logs\single_gpu_pretrained_eval_smoke_timeout.log") {
  Select-String -Path "repro_logs\single_gpu_pretrained_eval_smoke_timeout.log" -Pattern "Loading state dict|yaml_file:BDDX/testing|eval smoke timeout|it/s" | Select-Object -Last 20
}
