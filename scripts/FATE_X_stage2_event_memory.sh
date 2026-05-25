#!/usr/bin/env bash
set -euo pipefail
python -m fate_x.engine.smoke_fate_x_forward \
  --output .background_runs/fate_x_stage2_event_memory.json \
  --fate_x_enabled \
  --video_token_reducer none \
  --temporal_evidence_memory queries
