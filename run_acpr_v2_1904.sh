#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
mkdir -p '/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_1904/train'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:";$"LD_LIBRARY_PATH
export PYTHONPATH=.
nohup /opt/conda/envs/adapt/bin/python -m fate_x.engine.train_acpr_flowcal_v2 \
  --config configs/acpr_flowcal_v2_bddx_32f_224.yaml \
  --output_dir '/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_1904/train' \
  --device cuda \
  --epochs 15 \
  --batch_size 8 \
  --num_workers 6 \
  --gradient_accumulation_steps 4 \
  --resume /mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_staged_from_v2seed_b8w6_20260622_1617/train/checkpoint_best_joint.pth \
  --baseline_text_sum 2.012732295950884 \
  > '/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_1904/train_console.log' 2>&1 &
echo $! > '/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_1904/train.pid'
echo "started_at=$(date -Is) pid=$(cat '/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_1904/train.pid')" > '/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_1904/launcher_status.txt'
