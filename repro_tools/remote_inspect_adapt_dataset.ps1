$ErrorActionPreference = "Continue"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
$data = "E:\sbw\ADAPT_repro\ADAPT\ADAPT_PREPROCESSED_DATASET"

Write-Host "---REPO---"
Set-Location $repo
git rev-parse --short HEAD
git status --short

Write-Host "---README DATA REFERENCES---"
Select-String -Path "$repo\README.md" -Pattern "dataset|datasets|BDDX|tsv|checkpoint|pretrain|model|GPU|train|eval|test" -CaseSensitive:$false | Select-Object -First 120 | ForEach-Object { $_.Line }

Write-Host "---TOP TREE---"
Get-ChildItem -LiteralPath $repo -Force | Select-Object Mode,Length,Name | Format-Table -AutoSize

Write-Host "---DOWNLOADED DATA TOP---"
if (Test-Path $data) {
  Get-ChildItem -LiteralPath $data -Force | Select-Object Mode,Length,Name | Format-Table -AutoSize
  Write-Host "---DOWNLOADED DATA RECURSIVE SUMMARY---"
  Get-ChildItem -LiteralPath $data -Recurse -File | Group-Object Extension | Sort-Object Count -Descending | Select-Object Count,Name | Format-Table -AutoSize
  Write-Host "---LIKELY TSV/JSON/PKL/CKPT FILES---"
  Get-ChildItem -LiteralPath $data -Recurse -File | Where-Object { $_.Extension -match '\.tsv|\.json|\.pkl|\.pt|\.pth|\.bin|\.csv|\.txt|\.yaml|\.yml' } | Select-Object FullName,Length | Sort-Object FullName | Format-Table -AutoSize
} else {
  Write-Host "missing data dir: $data"
}

Write-Host "---TRAIN/EVAL SCRIPTS---"
Get-ChildItem -LiteralPath $repo -Recurse -File | Where-Object { $_.Name -match 'train|test|eval|infer|run|\.sh$|\.py$' } | Select-Object FullName | Sort-Object FullName | Format-Table -AutoSize

Write-Host "---PATH REFERENCES IN CODE---"
Get-ChildItem -LiteralPath $repo -Recurse -File -Include *.py,*.sh,*.json,*.yaml,*.yml,*.md | Select-String -Pattern "datasets|datasets_part|datasets-part|BDDX|/root|/home|checkpoint|ckpt|CUDA_VISIBLE_DEVICES|nproc|distributed|deepspeed|torchrun" -CaseSensitive:$false | Select-Object Path,LineNumber,Line | Format-List
