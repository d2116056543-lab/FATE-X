#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
python -u -m fate_x.engine.supervise_flowtrace_foreground --config configs/flowtrace_pmt_v1_bddx_32f_224.yaml --require_review_pass
