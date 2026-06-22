#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
export PYTHONPATH=.
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
RUN_DIR=/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_wsl_foreground_b4w4_resume_evalfix_20260622_014411/train
CKPT=${RUN_DIR}/checkpoint_latest.pth
LOG=${RUN_DIR}/resume_after_controlfix_console.log
STAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[${STAMP}] resume_after_controlfix start ckpt=${CKPT}" >> "${LOG}"
/opt/conda/envs/adapt/bin/python -m fate_x.engine.train_acpr_flowcal_v2 \
  --config configs/acpr_flowcal_v2_bddx_32f_224.yaml \
  --output_dir "${RUN_DIR}" \
  --device cuda \
  --epochs 15 \
  --batch_size 4 \
  --num_workers 4 \
  --gradient_accumulation_steps 16 \
  --resume "${CKPT}" \
  >> "${LOG}" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] resume_after_controlfix exit code $?" >> "${LOG}"
