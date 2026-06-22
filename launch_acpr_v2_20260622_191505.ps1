$ErrorActionPreference='Continue'
$run='E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191505'
New-Item -ItemType Directory -Force -Path "$run" | Out-Null
Set-Location 'E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree'
"START $(Get-Date -Format o)" | Out-File -FilePath "$run\task_status.log" -Encoding utf8 -Append
& wsl.exe -d ADAPT-Ubuntu -- /bin/bash '/mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree/run_acpr_v2_20260622_191505.sh' > "$run\train_stdout.log" 2> "$run\train_stderr.log"
$code=$LASTEXITCODE
"EXIT $(Get-Date -Format o) code=$code" | Out-File -FilePath "$run\task_status.log" -Encoding utf8 -Append
exit $code
