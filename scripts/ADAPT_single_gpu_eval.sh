#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EVAL_DIR="${EVAL_DIR:-checkpoints/basemodel/checkpoints/}"
DATA_DIR="${DATA_DIR:-datasets}"
VAL_YAML="${VAL_YAML:-BDDX/testing_32frames.yaml}"

python src/tasks/run_adapt.py \
  --data_dir "$DATA_DIR" \
  --val_yaml "$VAL_YAML" \
  --do_eval true \
  --do_train false \
  --eval_model_dir "$EVAL_DIR" \
  --per_gpu_eval_batch_size "${PER_GPU_EVAL_BATCH_SIZE:-4}" \
  --max_num_frames "${MAX_NUM_FRAMES:-32}"
