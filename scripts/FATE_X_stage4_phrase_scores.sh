#!/usr/bin/env bash
set -euo pipefail
python -m fate_x.engine.generate_decoder_phrase_scores_from_model \
  --replay_decoder_jsonl "${1:?decoder replay jsonl required}" \
  --output_jsonl .background_runs/fate_x_phrase_scores.jsonl \
  --summary_json .background_runs/fate_x_phrase_scores_summary.json \
  --max_samples "${2:-32}" \
  --beam_size 1 \
  --mask_strategy zero \
  --topk_ratio 0.10
