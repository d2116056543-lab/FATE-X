set -euo pipefail
printf '%s\n' '---before kill---'
ps -eo pid,ppid,etime,stat,cmd | grep -E 'acpr_flowcal_v2_wsl_debug_foreground|train_acpr_flowcal_v2' | grep -v grep || true
PIDS=$(ps -eo pid,cmd | grep -E 'acpr_flowcal_v2_wsl_debug_foreground|train_acpr_flowcal_v2' | grep -v grep | awk '{print $1}' || true)
if [ -n "${PIDS:-}" ]; then
  kill $PIDS || true
  sleep 3
  PIDS2=$(ps -eo pid,cmd | grep -E 'acpr_flowcal_v2_wsl_debug_foreground|train_acpr_flowcal_v2' | grep -v grep | awk '{print $1}' || true)
  if [ -n "${PIDS2:-}" ]; then kill -9 $PIDS2 || true; fi
fi
printf '%s\n' '---after kill---'
ps -eo pid,ppid,etime,stat,cmd | grep -E 'acpr_flowcal_v2_wsl_debug_foreground|train_acpr_flowcal_v2' | grep -v grep || true
printf '%s\n' '---gpu---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true