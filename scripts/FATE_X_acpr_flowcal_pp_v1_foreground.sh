#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml}"
OUT="${2:-.background_runs/acpr_flowcal_pp_v1_smoke}"
DEVICE="${3:-cuda}"
echo "ACPR foreground runner attached to this console."
python -m fate_x.engine.audit_acpr_flowcal_pp --config "$CONFIG" --output_dir "$OUT/preflight" --device "$DEVICE"
python -m fate_x.engine.train_acpr_flowcal_pp --config "$CONFIG" --output_dir "$OUT/train" --device "$DEVICE" --epochs 1 --max_steps 8 --batch_size 1 --beam_size 1
