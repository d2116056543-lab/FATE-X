#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt

mkdir -p repro_logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BATCH="${PER_GPU_TRAIN_BATCH_SIZE:-4}"
ACCUM="${GRADIENT_ACCUMULATION_STEPS:-16}"
TIMEOUT="${SMOKE_TIMEOUT_SECONDS:-180}"
OUT="./output/repro_single_gpu/mem_probe_b${BATCH}_a${ACCUM}"
LOG="repro_logs/single_gpu_mem_probe_b${BATCH}_a${ACCUM}.log"
GPU_LOG="repro_logs/single_gpu_mem_probe_b${BATCH}_a${ACCUM}_nvidia_smi.csv"

rm -f "$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$GPU_LOG" || true
    sleep 1
  done
) &
mon_pid=$!

set +e
OUTPUT_DIR="$OUT" \
NUM_TRAIN_EPOCHS=1 \
PER_GPU_TRAIN_BATCH_SIZE="$BATCH" \
PER_GPU_EVAL_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS="$ACCUM" \
timeout "${TIMEOUT}s" bash scripts/ADAPT_single_gpu_multitask.sh 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

kill "$mon_pid" >/dev/null 2>&1 || true
wait "$mon_pid" 2>/dev/null || true

python - "$GPU_LOG" <<'PY' | tee -a "$LOG"
import csv, sys
path=sys.argv[1]
vals=[]
try:
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    vals.append(int(row[1].strip()))
                except Exception:
                    pass
except FileNotFoundError:
    pass
print(f"nvidia_smi_samples={len(vals)}")
print(f"nvidia_smi_max_memory_mib={max(vals) if vals else 'NA'}")
PY

if [[ "$status" -eq 124 ]]; then
  echo "memory probe timeout reached; logs recorded" | tee -a "$LOG"
  exit 0
fi

exit "$status"
