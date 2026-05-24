#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

LOG_DIR="${LOG_DIR:-repro_logs}"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE=1

SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-180}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/repro_single_gpu/smoke_multitask_timeout}"

set +e
timeout "$SMOKE_TIMEOUT_SECONDS" bash scripts/ADAPT_single_gpu_smoke.sh 2>&1 | tee "$LOG_DIR/single_gpu_smoke_timeout.log"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -eq 124 ]]; then
  echo "ADAPT smoke timeout reached after ${SMOKE_TIMEOUT_SECONDS}s; inspect log for batch-proof lines." | tee -a "$LOG_DIR/single_gpu_smoke_timeout.log"
  exit 0
fi

exit "$status"
