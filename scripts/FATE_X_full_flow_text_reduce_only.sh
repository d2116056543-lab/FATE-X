#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON:-python}"
OUTPUT_DIR="${FATE_X_OUTPUT_DIR:-.background_runs/fate_x_full_text_reduce_only}"
DATA_DIR="${FATE_X_DATA_DIR:-datasets_part}"
TRAIN_YAML="${FATE_X_TRAIN_YAML:-BDDX/training_32frames.yaml}"
VAL_YAML="${FATE_X_VAL_YAML:-BDDX/testing_32frames.yaml}"
CHECKPOINT="${FATE_X_CHECKPOINT:-checkpoints/basemodel/checkpoints}"
EPOCHS="${FATE_X_EPOCHS:-40}"
BATCH_SIZE="${FATE_X_BATCH_SIZE:-2}"
GRAD_ACCUM="${FATE_X_GRAD_ACCUM:-32}"
COMMIT="$(git rev-parse HEAD)"

echo "FATE-X full-flow text-reduce-only template"
echo "git_commit=${COMMIT}"
echo "repo=${REPO_DIR}"
echo "data_dir=${DATA_DIR}"
echo "train_yaml=${TRAIN_YAML}"
echo "val_yaml=${VAL_YAML}"
echo "checkpoint=${CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"
echo "per_gpu_train_batch_size=${BATCH_SIZE}"
echo "gradient_accumulation_steps=${GRAD_ACCUM}"
echo "effective_batch_size=64"
echo "reducer=per_frame_topk_merge memory=queries text_reduce_only=true reduce_control=false learn_mask=false"

exec "$PYTHON_BIN" -m src.tasks.run_adapt \
  --do_train \
  --do_eval \
  --data_dir "$DATA_DIR" \
  --train_yaml "$TRAIN_YAML" \
  --val_yaml "$VAL_YAML" \
  --eval_model_dir "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$EPOCHS" \
  --per_gpu_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate 0.0002 \
  --backbone_coef_lr 0.05 \
  --reference_effective_batch 64 \
  --base_learning_rate_at_reference_batch 0.0002 \
  --auto_scale_lr false \
  --loss_sensor_w 0.05 \
  --max_num_frames 32 \
  --fate_x_enabled true \
  --video_token_reducer per_frame_topk_merge \
  --temporal_evidence_memory queries \
  --fate_x_text_reduce_only true \
  --fate_x_reduce_control false \
  --learn_mask_enabled false \
  --loss_sparse_w 0
