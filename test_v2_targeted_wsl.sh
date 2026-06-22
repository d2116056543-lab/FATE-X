set -euo pipefail
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
cd /mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
/opt/conda/envs/adapt/bin/python -m pytest tests/acpr_flowcal_v2/test_v2_training_monitoring_and_flow_metrics.py -q