$ErrorActionPreference = "Stop"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
$pre = Join-Path $repo "ADAPT_PREPROCESSED_DATASET"

Write-Host "---STOP STALE LINK SCRIPT---"
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "remote_link_adapt_preprocessed_dataset.ps1" } |
  ForEach-Object {
    Write-Host "stopping $($_.ProcessId) $($_.Name)"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }

function Rmdir-Link($path) {
  if (Test-Path -LiteralPath $path) {
    cmd /c "rmdir `"$path`""
  }
}

function MkJunction($link, $target) {
  if (-not (Test-Path -LiteralPath $target)) {
    throw "missing target $target"
  }
  Rmdir-Link $link
  $parent = Split-Path -Parent $link
  if (-not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
  }
  cmd /c "mklink /J `"$link`" `"$target`""
}

foreach ($name in @("BDDX", "BDDX_des", "BDDX_exp")) {
  MkJunction (Join-Path $repo "datasets\$name") (Join-Path $pre "datasets\$name")
}
MkJunction (Join-Path $repo "datasets_part") (Join-Path $pre "datasets_part")

Write-Host "---VERIFY WINDOWS PATHS---"
foreach ($p in @(
  "$repo\datasets\BDDX\training_32frames.yaml",
  "$repo\datasets\BDDX_des\training_32frames.yaml",
  "$repo\datasets\BDDX_exp\training_32frames.yaml",
  "$repo\datasets_part\BDDX\training_32frames.yaml",
  "$repo\datasets_part\BDDX\testing_32frames.yaml"
)) {
  if (-not (Test-Path -LiteralPath $p)) { throw "missing $p" }
  Write-Host "OK $p"
}

Write-Host "---LINKS AFTER---"
cmd /c dir /AL "$repo"
cmd /c dir /AL "$repo\datasets"
