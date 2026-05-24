$ErrorActionPreference = "Continue"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
Set-Location $repo

Write-Host "---ROOT DATASETS REAL CONTENT---"
Get-ChildItem -LiteralPath "$repo\datasets" -Force | Select-Object Mode,Length,FullName,Target | Format-List
Write-Host "---F ASSETS TOP---"
foreach ($p in @("F:\sbw_adapt_assets","F:\sbw_adapt_assets\datasets","F:\sbw_adapt_assets\models","F:\sbw_adapt_assets\checkpoints")) {
  Write-Host "---$p---"
  if (Test-Path $p) {
    Get-ChildItem -LiteralPath $p -Force | Select-Object Mode,Length,Name | Format-Table -AutoSize
  } else {
    Write-Host "missing"
  }
}

Write-Host "---DOWNLOADED READMES---"
Get-Content "$repo\ADAPT_PREPROCESSED_DATASET\Readme.txt" -Raw -ErrorAction SilentlyContinue

Write-Host "---MODEL/CHECKPOINT FILES---"
foreach ($p in @("$repo\models","$repo\checkpoints","F:\sbw_adapt_assets\models","F:\sbw_adapt_assets\checkpoints","$repo\ADAPT_PREPROCESSED_DATASET")) {
  Write-Host "---$p---"
  if (Test-Path $p) {
    Get-ChildItem -LiteralPath $p -Recurse -File -ErrorAction SilentlyContinue |
      Where-Object { $_.Extension -match '\.pt|\.pth|\.bin|\.json|\.txt|\.model|\.vocab|\.yaml|\.yml' } |
      Select-Object -First 120 FullName,Length |
      Format-Table -AutoSize
  } else {
    Write-Host "missing"
  }
}
