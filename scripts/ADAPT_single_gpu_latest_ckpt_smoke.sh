#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

mkdir -p repro_logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set +e
OUTPUT_DIR="${OUTPUT_DIR:-./output/repro_single_gpu/latest_ckpt_smoke}" \
NUM_TRAIN_EPOCHS=1 \
PER_GPU_TRAIN_BATCH_SIZE="${PER_GPU_TRAIN_BATCH_SIZE:-2}" \
PER_GPU_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}" \
timeout "${SMOKE_TIMEOUT_SECONDS:-180}s" bash scripts/ADAPT_single_gpu_multitask.sh \
  2>&1 | tee repro_logs/single_gpu_latest_ckpt_smoke.log
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -eq 124 ]]; then
  echo "latest checkpoint smoke timeout reached; inspect checkpoint_latest." | tee -a repro_logs/single_gpu_latest_ckpt_smoke.log
  exit 0
fi

exit "$status"
