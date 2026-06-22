$ErrorActionPreference = 'Stop'
$runWin = 'E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_wsl_startprocess_lf_b4w4_full_20260622_000342'
$runWsl = '/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_wsl_startprocess_lf_b4w4_full_20260622_000342'
$p = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d','ADAPT-Ubuntu','bash',"$runWsl/run_train_logged.sh") -WindowStyle Hidden -PassThru
[IO.File]::WriteAllText((Join-Path $runWin 'windows_wsl.pid'), [string]$p.Id, [Text.Encoding]::ASCII)
[IO.File]::WriteAllText('E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_latest_run_win.txt', $runWin, (New-Object Text.UTF8Encoding($false)))
[IO.File]::WriteAllText('E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_latest_run.txt', $runWsl, (New-Object Text.UTF8Encoding($false)))
Write-Host "RUN_WIN=$runWin"
Write-Host "RUN_WSL=$runWsl"
Write-Host "WINDOWS_WSL_PID=$($p.Id)"
Start-Sleep -Seconds 25
Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq $p.Id -or ($_.CommandLine -match [regex]::Escape($runWsl)) } | Select-Object ProcessId,Name,CommandLine | Format-List
if (Test-Path (Join-Path $runWin 'train.log')) { Get-Content (Join-Path $runWin 'train.log') -Tail 100 } else { Write-Host 'train_log_missing' }
wsl -d ADAPT-Ubuntu bash -lc "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true"