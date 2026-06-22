#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m fate_x.engine.supervise_acpr_flowcal_v2_foreground \
  --config configs/acpr_flowcal_v2_bddx_32f_224.yaml \
  --output_dir "${1:-.background_runs/acpr_flowcal_v2_linux}" \
  --device cuda \
  --batch_size 4 \
  --num_workers 4 \
  --gradient_accumulation_steps 8 \
  --epochs 15
