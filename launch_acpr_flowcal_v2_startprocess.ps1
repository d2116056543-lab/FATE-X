$ErrorActionPreference = 'Stop'
Write-Host '---existing relevant processes---'
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_acpr_flowcal_v2|acpr_flowcal_v2_wsl_startprocess' } | Select-Object ProcessId,Name,CommandLine | Format-List
$repoWin = 'E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree'
$repoWsl = '/mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$runWin = "E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_wsl_startprocess_b4w4_full_$ts"
$runWsl = "/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_wsl_startprocess_b4w4_full_$ts"
New-Item -ItemType Directory -Force -Path (Join-Path $runWin 'train') | Out-Null
$bash = @"
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:`${LD_LIBRARY_PATH:-}
export PYTHONFAULTHANDLER=1
RUN='$runWsl'
cd '$repoWsl'
exec > "`$RUN/train.log" 2>&1
echo "ACPR_FLOWCAL_V2_START `$(date -Is)"
echo "RUN=`$RUN"
echo "CMD batch_size=4 num_workers=4 gradient_accumulation_steps=16 epochs=15"
set +e
/opt/conda/envs/adapt/bin/python -u -m fate_x.engine.train_acpr_flowcal_v2 \
  --config configs/acpr_flowcal_v2_bddx_32f_224.yaml \
  --output_dir "`$RUN/train" \
  --device cuda \
  --epochs 15 \
  --batch_size 4 \
  --num_workers 4 \
  --gradient_accumulation_steps 16
EC=`$?
set -e
echo "ACPR_FLOWCAL_V2_EXIT_CODE=`$EC"
echo "ACPR_FLOWCAL_V2_END `$(date -Is)"
echo "`$EC" > "`$RUN/exit_code.txt"
exit "`$EC"
"@
[IO.File]::WriteAllText((Join-Path $runWin 'run_train_logged.sh'), $bash, (New-Object Text.UTF8Encoding($false)))
$envObj = [ordered]@{run_dir=$runWin; run_dir_wsl=$runWsl; repo=$repoWin; repo_wsl=$repoWsl; launcher='Start-Process wsl.exe'; distro='ADAPT-Ubuntu'; python='/opt/conda/envs/adapt/bin/python'; batch_size=4; num_workers=4; gradient_accumulation_steps=16; epochs=15; device='cuda'; launch_time=(Get-Date).ToString('o')}
[IO.File]::WriteAllText((Join-Path $runWin 'run_env.json'), ($envObj | ConvertTo-Json -Depth 4), (New-Object Text.UTF8Encoding($false)))
$wslScript = "$runWsl/run_train_logged.sh"
$p = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d','ADAPT-Ubuntu','bash',$wslScript) -WindowStyle Hidden -PassThru
[IO.File]::WriteAllText((Join-Path $runWin 'windows_wsl.pid'), [string]$p.Id, [Text.Encoding]::ASCII)
[IO.File]::WriteAllText('E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_latest_run_win.txt', $runWin, (New-Object Text.UTF8Encoding($false)))
[IO.File]::WriteAllText('E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_latest_run.txt', $runWsl, (New-Object Text.UTF8Encoding($false)))
Write-Host "RUN_WIN=$runWin"
Write-Host "RUN_WSL=$runWsl"
Write-Host "WINDOWS_WSL_PID=$($p.Id)"
Start-Sleep -Seconds 20
Write-Host '---process windows---'
Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq $p.Id -or ($_.CommandLine -match [regex]::Escape($runWsl)) -or ($_.CommandLine -match 'train_acpr_flowcal_v2') } | Select-Object ProcessId,Name,CommandLine | Format-List
Write-Host '---train log tail---'
$log = Join-Path $runWin 'train.log'
if (Test-Path $log) { Get-Content $log -Tail 80 } else { Write-Host 'train_log_missing' }
Write-Host '---gpu---'
wsl -d ADAPT-Ubuntu bash -lc "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true"
