#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/acpr_dynflow_v1_bddx_32f_224.yaml}"
OUT="${2:-/mnt/e/sbw/FATE_Drive/active_runs/acpr_dynflow_v1_formal}"
python -m fate_x.engine.supervise_acpr_dynflow_foreground --config "$CONFIG" --output_dir "$OUT" --device cuda

