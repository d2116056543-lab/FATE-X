#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml}"
python -m fate_x.engine.supervise_acpr_dynflow_swin_foreground --config "$CONFIG" --require_review_pass
