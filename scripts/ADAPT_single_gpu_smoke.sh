#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE=1

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port="${MASTER_PORT:-45979}" src/tasks/run_adapt.py \
  --config src/configs/VidSwinBert/BDDX_multi_default.json \
  --data_dir datasets_part \
  --train_yaml BDDX/training_32frames.yaml \
  --val_yaml BDDX/testing_32frames.yaml \
  --per_gpu_train_batch_size "${PER_GPU_TRAIN_BATCH_SIZE:-1}" \
  --per_gpu_eval_batch_size "${PER_GPU_EVAL_BATCH_SIZE:-1}" \
  --limited_samples "${LIMITED_SAMPLES:-8}" \
  --num_train_epochs 1 \
  --learning_rate 0.0002 \
  --max_num_frames 32 \
  --pretrained_2d 0 \
  --backbone_coef_lr 0.05 \
  --mask_prob 0.5 \
  --max_masked_token 45 \
  --zero_opt_stage 1 \
  --mixed_precision_method deepspeed \
  --deepspeed_fp16 \
  --gradient_accumulation_steps 1 \
  --learn_mask_enabled \
  --loss_sparse_w 0.1 \
  --use_sep_cap \
  --multitask \
  --signal_types course speed \
  --loss_sensor_w 0.05 \
  --max_grad_norm 1 \
  --debug \
  --output_dir "${OUTPUT_DIR:-./output/repro_single_gpu/smoke_multitask}"
