$ErrorActionPreference = "Continue"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
$data = "E:\sbw\ADAPT_repro\ADAPT\ADAPT_PREPROCESSED_DATASET"

Write-Host "---RUNNING INSPECT PROCESSES---"
Get-Process | Where-Object { $_.ProcessName -match 'powershell|findstr|Select-String' } | Select-Object Id,ProcessName,CPU | Format-Table -AutoSize

Write-Host "---REPO---"
Set-Location $repo
git rev-parse --short HEAD
git status --short

Write-Host "---README DATA REFERENCES---"
if (Test-Path "$repo\README.md") {
  Select-String -Path "$repo\README.md" -Pattern "dataset|datasets|BDDX|tsv|checkpoint|pretrain|model|GPU|train|eval|test" -CaseSensitive:$false |
    Select-Object -First 160 |
    ForEach-Object { "{0}:{1}" -f $_.LineNumber, $_.Line }
}

Write-Host "---REPO TOP TREE---"
Get-ChildItem -LiteralPath $repo -Force | Select-Object Mode,Length,Name | Format-Table -AutoSize

Write-Host "---DATA TOP TREE---"
if (Test-Path $data) {
  Get-ChildItem -LiteralPath $data -Force | Select-Object Mode,Length,Name | Format-Table -AutoSize
  foreach ($sub in @("datasets","datasets-part","datasets_part","checkpoints","pretrained","models")) {
    $p = Join-Path $data $sub
    if (Test-Path $p) {
      Write-Host "---$sub TOP---"
      Get-ChildItem -LiteralPath $p -Force | Select-Object Mode,Length,Name | Format-Table -AutoSize
      Write-Host "---$sub COUNTS---"
      $files = Get-ChildItem -LiteralPath $p -Recurse -File -ErrorAction SilentlyContinue
      $bytes = ($files | Measure-Object -Property Length -Sum).Sum
      "file_count={0} bytes={1}" -f @($files).Count, $bytes
      $files | Where-Object { $_.Extension -match '\.tsv|\.json|\.pkl|\.pt|\.pth|\.bin|\.csv|\.txt|\.yaml|\.yml' } |
        Select-Object -First 80 FullName,Length |
        Format-Table -AutoSize
    }
  }
} else {
  Write-Host "missing data dir: $data"
}

Write-Host "---SCRIPT FILES TOP---"
Get-ChildItem -LiteralPath $repo -Recurse -File -Include *.sh,*.py,*.json,*.yaml,*.yml -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch "ADAPT_PREPROCESSED_DATASET|\\.git|__pycache__" } |
  Select-Object -First 200 FullName |
  Format-Table -AutoSize
