$ErrorActionPreference = "Continue"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
Set-Location $repo
function Print-Lines($path, $start, $end) {
  Write-Host "---$path $start-$end---"
  $i=0
  Get-Content $path | ForEach-Object {
    $i++
    if ($i -ge $start -and $i -le $end) { "{0}: {1}" -f $i, $_ }
  }
}
Print-Lines "src\datasets\vl_dataloader.py" 1 80
Print-Lines "src\datasets\vision_language_tsv.py" 1 120
Print-Lines "src\configs\config.py" 330 390
Print-Lines "src\tasks\run_adapt.py" 880 920
