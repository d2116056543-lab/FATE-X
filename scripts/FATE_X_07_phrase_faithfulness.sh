#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m fate_x.engine.eval_phrase_faithfulness \
  --predictions_jsonl "${PREDICTIONS_JSONL:?set PREDICTIONS_JSONL}" \
  --output "${OUTPUT:-outputs/fate_x/phrase_faithfulness.json}"
