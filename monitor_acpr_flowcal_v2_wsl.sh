set -euo pipefail
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
RUN=$(cat /mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_latest_run.txt)
printf 'RUN=%s\n' "$RUN"
printf '%s\n' '---process---'
PID=$(cat "$RUN/train.pid")
ps -p "$PID" -o pid,ppid,etime,cmd || true
ps -eo pid,ppid,etime,cmd | grep -E 'train_acpr_flowcal_v2|python|run_train.sh' | grep -v grep || true
printf '%s\n' '---gpu---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
printf '%s\n' '---train.log tail---'
tail -120 "$RUN/train.log" || true
printf '%s\n' '---manifest---'
if [ -f "$RUN/train/run_manifest.json" ]; then cat "$RUN/train/run_manifest.json"; else echo missing; fi
printf '%s\n' '---progress---'
if [ -f "$RUN/train/train_progress.jsonl" ]; then tail -20 "$RUN/train/train_progress.jsonl"; else echo missing; fi
printf '%s\n' '---metrics---'
if [ -f "$RUN/train/metrics_summary.jsonl" ]; then tail -5 "$RUN/train/metrics_summary.jsonl"; else echo missing; fi
printf '%s\n' '---checkpoints---'
ls -lh "$RUN/train"/checkpoint*.pth 2>/dev/null || true