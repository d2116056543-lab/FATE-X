#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/sbw/ADAPT_repro/ADAPT
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  echo "Set CHECKPOINT_DIR to checkpoint_latest or checkpoint_best before running." >&2
  exit 2
fi

mkdir -p repro_logs
RUN_NAME="${RUN_NAME:-adapt_resume_b4a16_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="repro_logs/${RUN_NAME}.log"
exec > >(tee -a "$LOG_PATH") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output/repro_single_gpu/${RUN_NAME}}"
export PER_GPU_TRAIN_BATCH_SIZE="${PER_GPU_TRAIN_BATCH_SIZE:-4}"
export PER_GPU_EVAL_BATCH_SIZE="${PER_GPU_EVAL_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-40}"

printf 'ADAPT resume single-GPU run started at %s\n' "$(date -Is)"
printf 'CHECKPOINT_DIR=%s\n' "$CHECKPOINT_DIR"
printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"

exec python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port="${MASTER_PORT:-45979}" src/tasks/run_adapt.py \
  --config src/configs/VidSwinBert/BDDX_multi_default.json \
  --data_dir datasets_part \
  --train_yaml BDDX/training_32frames.yaml \
  --val_yaml BDDX/testing_32frames.yaml \
  --per_gpu_train_batch_size "${PER_GPU_TRAIN_BATCH_SIZE}" \
  --per_gpu_eval_batch_size "${PER_GPU_EVAL_BATCH_SIZE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate 0.0002 \
  --max_num_frames 32 \
  --pretrained_2d 0 \
  --backbone_coef_lr 0.05 \
  --mask_prob 0.5 \
  --max_masked_token 45 \
  --zero_opt_stage 1 \
  --mixed_precision_method deepspeed \
  --deepspeed_fp16 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learn_mask_enabled \
  --loss_sparse_w 0.1 \
  --use_sep_cap \
  --multitask \
  --signal_types course speed \
  --loss_sensor_w 0.05 \
  --max_grad_norm 1 \
  --output_dir "${OUTPUT_DIR}" \
  --resume_repro_checkpoint_dir "${CHECKPOINT_DIR}"
