#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "FATE-X training wrapper. Requires ADAPT run_adapt.py support for FATE flags."
echo "Requested flags: --fate_x_enabled --video_token_reducer topk_merge --temporal_evidence_memory queries --phrase_faithfulness_enabled"
bash scripts/ADAPT_single_gpu_multitask.sh --fate_x_enabled --video_token_reducer topk_merge --temporal_evidence_memory queries --phrase_faithfulness_enabled
