#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export EVAL_DIR="${EVAL_DIR:-checkpoints/basemodel/checkpoints/}"
export DATA_DIR="${DATA_DIR:-datasets}"
export VAL_YAML="${VAL_YAML:-BDDX/testing_32frames.yaml}"
export PER_GPU_EVAL_BATCH_SIZE="${PER_GPU_EVAL_BATCH_SIZE:-4}"
if [[ -x scripts/ADAPT_single_gpu_eval.sh || -f scripts/ADAPT_single_gpu_eval.sh ]]; then
  bash scripts/ADAPT_single_gpu_eval.sh
else
  echo "Missing scripts/ADAPT_single_gpu_eval.sh" >&2
  exit 2
fi
