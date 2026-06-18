$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree"
E:\Anaconda\envs\sbw39\python.exe -u -m fate_x.engine.supervise_flowtrace_foreground --config configs/flowtrace_pmt_v1_bddx_32f_224.yaml --require_review_pass
