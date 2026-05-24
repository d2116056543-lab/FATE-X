$ErrorActionPreference = "Continue"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
Set-Location $repo

Write-Host "---SCRIPT CONTENTS---"
foreach ($f in @(
  "scripts\BDDX_multitask.sh",
  "scripts\BDDX_test.sh",
  "scripts\BDDX_only_caption.sh",
  "scripts\BDDX_only_signal.sh",
  "scripts\other_scripts\BDDX_multitask_des.sh",
  "scripts\other_scripts\BDDX_multitask_exp.sh"
)) {
  Write-Host "---$f---"
  if (Test-Path $f) { Get-Content $f -Raw } else { Write-Host "missing" }
}

Write-Host "---CONFIG JSONS---"
foreach ($f in @(
  "src\configs\VidSwinBert\BDDX_multi_default.json",
  "src\configs\VidSwinBert\BDDX_two_default.json",
  "src\configs\VidSwinBert\BDDX_8frm_default.json"
)) {
  Write-Host "---$f---"
  if (Test-Path $f) { Get-Content $f -Raw } else { Write-Host "missing" }
}

Write-Host "---YAML HEADS---"
foreach ($f in @(
  "datasets\BDDX\training_32frames.yaml",
  "datasets\BDDX\validation_32frames.yaml",
  "datasets\BDDX\testing_32frames.yaml",
  "ADAPT_PREPROCESSED_DATASET\datasets\BDDX\training_32frames.yaml",
  "ADAPT_PREPROCESSED_DATASET\datasets_part\BDDX\training_32frames.yaml"
)) {
  Write-Host "---$f---"
  if (Test-Path $f) { Get-Content $f -TotalCount 30 } else { Write-Host "missing" }
}

Write-Host "---LINK TARGETS---"
cmd /c dir /AL
cmd /c dir /AL datasets
cmd /c dir /AL models
cmd /c dir /AL checkpoints
