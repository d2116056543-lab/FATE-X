#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/sbw/ADAPT_repro/ADAPT
mkdir -p repro_logs

nohup bash scripts/ADAPT_single_gpu_full_b4a16_run.sh \
  > repro_logs/adapt_full_b4a16_nohup.out 2>&1 &
pid=$!

printf '%s\n' "$pid" > repro_logs/adapt_full_b4a16_linux_pid.txt
printf 'started_linux_pid=%s\n' "$pid"
printf 'started_at=%s\n' "$(date -Is)"
