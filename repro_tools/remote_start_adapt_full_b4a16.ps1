$ErrorActionPreference = "Stop"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
$runName = "adapt_full_b4a16_20260522_0036"
$outDir = "E:\sbw\ADAPT_repro\ADAPT\output\repro_single_gpu\$runName"
$logPath = "E:\sbw\ADAPT_repro\ADAPT\repro_logs\$runName.log"
$pidPath = "E:\sbw\ADAPT_repro\ADAPT\repro_logs\$runName.pid.txt"

Set-Location $repo
New-Item -ItemType Directory -Force -Path "repro_logs" | Out-Null

$arg = "-d ADAPT-Ubuntu -- bash -lc `"cd /mnt/e/sbw/ADAPT_repro/ADAPT && bash scripts/ADAPT_single_gpu_full_b4a16_run.sh >> repro_logs/$runName.log 2>&1`""
$proc = Start-Process -FilePath "wsl.exe" -ArgumentList $arg -WorkingDirectory $repo -WindowStyle Hidden -PassThru
"WindowsPID=$($proc.Id)`nRunName=$runName`nOutputDir=$outDir`nLog=$logPath" | Set-Content $pidPath -Encoding UTF8

Write-Host "started"
Get-Content $pidPath
