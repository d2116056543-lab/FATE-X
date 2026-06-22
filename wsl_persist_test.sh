set -euo pipefail
cd /mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
mkdir -p /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test
rm -f /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/*
(command -v tmux && echo tmux_ok || echo tmux_missing) > /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/capabilities.txt
(command -v screen && echo screen_ok || echo screen_missing) >> /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/capabilities.txt
(command -v setsid && echo setsid_ok || echo setsid_missing) >> /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/capabilities.txt
nohup bash -lc 'sleep 45; date -Is > /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/nohup_done.txt' > /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/nohup.log 2>&1 &
setsid bash -lc 'sleep 45; date -Is > /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/setsid_done.txt' > /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/setsid.log 2>&1 < /dev/null &
if command -v tmux >/dev/null 2>&1; then tmux new-session -d -s acpr_persist_test "bash -lc 'sleep 45; date -Is > /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/tmux_done.txt'"; fi
if command -v screen >/dev/null 2>&1; then screen -dmS acpr_persist_test bash -lc 'sleep 45; date -Is > /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/screen_done.txt'; fi
ps -eo pid,ppid,etime,cmd | grep -E 'wsl_persist_test|tmux|screen' | grep -v grep > /mnt/e/sbw/FATE_Drive/active_runs/wsl_persist_test/initial_ps.txt || true