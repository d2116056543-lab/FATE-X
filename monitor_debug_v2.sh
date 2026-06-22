set -euo pipefail
printf '%s\n' '---process---'
ps -eo pid,ppid,etime,stat,cmd | grep -E 'train_acpr_flowcal_v2|python' | grep -v grep || true
printf '%s\n' '---gpu---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
printf '%s\n' '---debug runs---'
RUN=$(ls -dt /mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_wsl_debug_foreground_* 2>/dev/null | head -1 || true)
echo "$RUN"
if [ -n "$RUN" ]; then
  find "$RUN" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort
  [ -f "$RUN/train/run_manifest.json" ] && cat "$RUN/train/run_manifest.json" || true
  [ -f "$RUN/train/train_progress.jsonl" ] && tail -20 "$RUN/train/train_progress.jsonl" || true
fi