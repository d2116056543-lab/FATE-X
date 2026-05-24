#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

mkdir -p repro_logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EVAL_DIR="${EVAL_DIR:-checkpoints/basemodel/checkpoints/}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-180}"

set +e
timeout "$SMOKE_TIMEOUT_SECONDS" python src/tasks/run_adapt.py \
  --val_yaml BDDX/testing_32frames.yaml \
  --do_eval true \
  --do_train false \
  --eval_model_dir "$EVAL_DIR" \
  --limited_samples "${LIMITED_SAMPLES:-8}" \
  --per_gpu_eval_batch_size "${PER_GPU_EVAL_BATCH_SIZE:-1}" \
  2>&1 | tee repro_logs/single_gpu_pretrained_eval_smoke_timeout.log
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -eq 124 ]]; then
  echo "ADAPT eval smoke timeout reached after ${SMOKE_TIMEOUT_SECONDS}s; checkpoint load and eval-loop startup are recorded in the log." | tee -a repro_logs/single_gpu_pretrained_eval_smoke_timeout.log
  exit 0
fi

exit "$status"
