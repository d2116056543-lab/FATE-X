#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/sbw/ADAPT_repro/ADAPT
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

mkdir -p repro_logs
RUN_NAME="${RUN_NAME:-adapt_full_b4a16_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="repro_logs/${RUN_NAME}.log"
exec > >(tee -a "$LOG_PATH") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/repro_single_gpu/${RUN_NAME}}"
export PER_GPU_TRAIN_BATCH_SIZE="${PER_GPU_TRAIN_BATCH_SIZE:-4}"
export PER_GPU_EVAL_BATCH_SIZE="${PER_GPU_EVAL_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-40}"

printf 'ADAPT full single-GPU run started at %s\n' "$(date -Is)"
printf 'RUN_NAME=%s\n' "$RUN_NAME"
printf 'LOG_PATH=%s\n' "$LOG_PATH"
printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'PER_GPU_TRAIN_BATCH_SIZE=%s\n' "$PER_GPU_TRAIN_BATCH_SIZE"
printf 'PER_GPU_EVAL_BATCH_SIZE=%s\n' "$PER_GPU_EVAL_BATCH_SIZE"
printf 'GRADIENT_ACCUMULATION_STEPS=%s\n' "$GRADIENT_ACCUMULATION_STEPS"
printf 'EFFECTIVE_BATCH=%s\n' "$((PER_GPU_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
printf 'NUM_TRAIN_EPOCHS=%s\n' "$NUM_TRAIN_EPOCHS"
printf 'GIT_HEAD=%s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

exec bash scripts/ADAPT_single_gpu_multitask.sh
