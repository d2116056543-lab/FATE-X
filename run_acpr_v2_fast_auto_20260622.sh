#!/usr/bin/env bash
set -uo pipefail
cd /mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
OUT=/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191958/train
RESUME=$OUT/checkpoint_latest.pth
BASELINE=2.012732295950884
RUN_ROOT=/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191958
FALLBACK_LOG=$RUN_ROOT/train_fast_fallback.log

run_one() {
  local bs="$1"
  local acc="$2"
  local nw="$3"
  local tag="$4"
  local log=$RUN_ROOT/train_fast_${tag}.log
  echo "[fast-launch] $(date -Is) tag=$tag batch_size=$bs grad_accum=$acc num_workers=$nw resume=$RESUME" >> "$log"
  /opt/conda/envs/adapt/bin/python -m fate_x.engine.train_acpr_flowcal_v2 \
    --config configs/acpr_flowcal_v2_bddx_32f_224.yaml \
    --output_dir "$OUT" \
    --device cuda \
    --epochs 15 \
    --batch_size "$bs" \
    --num_workers "$nw" \
    --gradient_accumulation_steps "$acc" \
    --resume "$RESUME" \
    --baseline_text_sum "$BASELINE" >> "$log" 2>&1
  local status=$?
  echo "[fast-launch] $(date -Is) tag=$tag exit_code=$status" >> "$log"
  echo "[fast-launch] $(date -Is) tag=$tag exit_code=$status" >> "$FALLBACK_LOG"
  return "$status"
}

if run_one 32 1 8 b32_acc1_w8; then
  exit 0
fi
echo "[fast-launch] batch32 failed, trying batch24" >> "$FALLBACK_LOG"

if run_one 24 1 8 b24_acc1_w8; then
  exit 0
fi
echo "[fast-launch] batch24 failed, trying batch16 accum2" >> "$FALLBACK_LOG"

if run_one 16 2 8 b16_acc2_w8; then
  exit 0
fi
echo "[fast-launch] batch16 failed, falling back to original batch8 accum4" >> "$FALLBACK_LOG"
run_one 8 4 6 b8_acc4_w6_recovery
