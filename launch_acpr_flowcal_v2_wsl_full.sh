set -euo pipefail
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
REPO=/mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
cd "$REPO"
TS=$(date +%Y%m%d_%H%M%S)
RUN=/mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_wsl_b4w4_full_${TS}
mkdir -p "$RUN/train"
cat > "$RUN/run_train.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
cd /mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
/opt/conda/envs/adapt/bin/python -u -m fate_x.engine.train_acpr_flowcal_v2 \
  --config configs/acpr_flowcal_v2_bddx_32f_224.yaml \
  --output_dir "$RUN_DIR/train" \
  --device cuda \
  --epochs 15 \
  --batch_size 4 \
  --num_workers 4 \
  --gradient_accumulation_steps 16
SH
chmod +x "$RUN/run_train.sh"
cat > "$RUN/run_env.json" <<JSON
{"run_dir":"$RUN","repo":"$REPO","python":"/opt/conda/envs/adapt/bin/python","batch_size":4,"num_workers":4,"gradient_accumulation_steps":16,"epochs":15,"device":"cuda","started_by":"codex","launch_time":"$(date -Is)"}
JSON
RUN_DIR="$RUN" nohup bash "$RUN/run_train.sh" > "$RUN/train.log" 2>&1 &
PID=$!
echo "$PID" > "$RUN/train.pid"
echo "$RUN" > /mnt/e/sbw/FATE_Drive/active_runs/acpr_flowcal_v2_latest_run.txt
printf 'RUN_DIR=%s\nPID=%s\n' "$RUN" "$PID"
sleep 5
printf '%s\n' '---process---'
ps -p "$PID" -o pid,ppid,etime,cmd || true
printf '%s\n' '---log head/tail---'
tail -80 "$RUN/train.log" || true
printf '%s\n' '---manifest---'
if [ -f "$RUN/train/run_manifest.json" ]; then cat "$RUN/train/run_manifest.json"; else echo 'manifest_not_yet_written'; fi
printf '%s\n' '---gpu---'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true