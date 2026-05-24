$ErrorActionPreference = "Stop"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
$pre = Join-Path $repo "ADAPT_PREPROCESSED_DATASET"

function Remove-ReparseOrDir($path) {
  if (Test-Path -LiteralPath $path) {
    $item = Get-Item -LiteralPath $path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $item.LinkType) {
      Remove-Item -LiteralPath $path -Force
    } else {
      throw "Refusing to remove non-link directory: $path"
    }
  }
}

function New-Junction($link, $target) {
  if (-not (Test-Path -LiteralPath $target)) {
    throw "Missing target: $target"
  }
  Remove-ReparseOrDir $link
  $parent = Split-Path -Parent $link
  if (-not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
  }
  cmd /c "mklink /J `"$link`" `"$target`"" | Write-Host
}

Set-Location $repo

# Preserve datasets\BDDX_raw. Replace only the processed BDDX links.
foreach ($name in @("BDDX", "BDDX_des", "BDDX_exp")) {
  New-Junction (Join-Path $repo "datasets\$name") (Join-Path $pre "datasets\$name")
}

New-Junction (Join-Path $repo "datasets_part") (Join-Path $pre "datasets_part")

Write-Host "---LINKS AFTER---"
cmd /c dir /AL "$repo"
cmd /c dir /AL "$repo\datasets"

Write-Host "---PATH CHECK---"
foreach ($p in @(
  "$repo\datasets\BDDX\training_32frames.yaml",
  "$repo\datasets\BDDX_des\training_32frames.yaml",
  "$repo\datasets\BDDX_exp\training_32frames.yaml",
  "$repo\datasets_part\BDDX\training_32frames.yaml",
  "$repo\datasets_part\BDDX\testing_32frames.yaml"
)) {
  if (Test-Path -LiteralPath $p) { Write-Host "OK $p" } else { throw "missing $p" }
}
