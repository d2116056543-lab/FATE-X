#!/usr/bin/env bash
set -euo pipefail
python -m fate_x.engine.smoke_fate_x_forward \
  --output .background_runs/fate_x_stage3_text_reducer.json \
  --fate_x_enabled \
  --video_token_reducer topk_merge \
  --temporal_evidence_memory queries \
  --fate_x_text_reduce_only true
