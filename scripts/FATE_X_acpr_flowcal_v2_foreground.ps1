param(
  [string]$OutputDir = ".background_runs\acpr_flowcal_v2_windows_forwarder"
)
$ErrorActionPreference = "Stop"
wsl -d ADAPT-Ubuntu -- bash -lc "cd /mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree && python -m fate_x.engine.supervise_acpr_flowcal_v2_foreground --config configs/acpr_flowcal_v2_bddx_32f_224.yaml --output_dir '$OutputDir' --device cuda --batch_size 4 --num_workers 4 --gradient_accumulation_steps 8 --epochs 15"
