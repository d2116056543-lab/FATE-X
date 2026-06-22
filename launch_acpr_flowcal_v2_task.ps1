$ErrorActionPreference='Continue'
$runWin='E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_wsl_task_b4w4_full_20260622_000802'
$runWsl='/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_wsl_task_b4w4_full_20260622_000802'
$taskName='acpr_flowcal_v2_20260622_000802'
[IO.File]::WriteAllText('E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_latest_run_win.txt', $runWin, (New-Object Text.UTF8Encoding($false)))
[IO.File]::WriteAllText('E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_latest_run.txt', $runWsl, (New-Object Text.UTF8Encoding($false)))
[IO.File]::WriteAllText((Join-Path $runWin 'task_name.txt'), $taskName, (New-Object Text.UTF8Encoding($false)))
$tr = 'C:\Windows\System32\wsl.exe -d ADAPT-Ubuntu bash ' + $runWsl + '/run_train_logged.sh'
$st = (Get-Date).AddMinutes(1).ToString('HH:mm')
schtasks /Delete /TN $taskName /F 2>$null | Out-Null
schtasks /Create /TN $taskName /SC ONCE /ST $st /TR $tr /F | Write-Host
schtasks /Run /TN $taskName | Write-Host
Write-Host "RUN_WIN=$runWin"
Write-Host "RUN_WSL=$runWsl"
Write-Host "TASK_NAME=$taskName"
Start-Sleep -Seconds 35
Write-Host '---task query---'
schtasks /Query /TN $taskName /V /FO LIST
Write-Host '---processes---'
Get-CimInstance Win32_Process | Where-Object { ($_.CommandLine -match [regex]::Escape($runWsl)) -or ($_.CommandLine -match 'train_acpr_flowcal_v2') -or ($_.Name -match 'wsl') } | Select-Object ProcessId,Name,CommandLine | Format-List
Write-Host '---train log---'
if (Test-Path (Join-Path $runWin 'train.log')) { Get-Content (Join-Path $runWin 'train.log') -Tail 120 } else { Write-Host 'train_log_missing' }
Write-Host '---gpu---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits