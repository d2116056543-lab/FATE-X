#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml}"
OUT="${2:-.background_runs/acpr_flowcal_pp_v1_formal_foreground}"
DEVICE="${3:-cuda}"
EPOCHS="${EPOCHS:-21}"
MAX_STEPS="${MAX_STEPS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
BEAM_SIZE="${BEAM_SIZE:-3}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-500}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"
REQUIRE_REVIEW_PASS="${REQUIRE_REVIEW_PASS:-0}"
REVIEW_PASS_DIR="${REVIEW_PASS_DIR:-}"
echo "ACPR foreground runner attached to this console."

if [[ "$REQUIRE_REVIEW_PASS" == "1" ]]; then
  if [[ -z "$REVIEW_PASS_DIR" ]]; then
    echo "Formal ACPR-FlowCal++ training is blocked: REQUIRE_REVIEW_PASS=1 needs REVIEW_PASS_DIR for the current clean pushed HEAD." >&2
    exit 1
  fi
  PASS_FILE="$REVIEW_PASS_DIR/REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt"
  if [[ ! -f "$PASS_FILE" ]]; then
    echo "Formal ACPR-FlowCal++ training is blocked: missing $PASS_FILE." >&2
    exit 1
  fi
  HEAD_SHA="$(git rev-parse HEAD)"
  if ! grep -Fq "$HEAD_SHA" "$PASS_FILE"; then
    echo "Formal ACPR-FlowCal++ training is blocked: $PASS_FILE is not bound to current HEAD $HEAD_SHA." >&2
    exit 1
  fi
fi

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  python -m fate_x.engine.probe_acpr_flowcal_memory --config "$CONFIG" --output_dir "$OUT/preflight" --device "$DEVICE"
  python -m fate_x.engine.run_acpr_flowcal_preflight_gates --config "$CONFIG" --output_dir "$OUT/preflight" --device "$DEVICE"
  python -m fate_x.engine.audit_acpr_flowcal_pp --config "$CONFIG" --output_dir "$OUT/preflight" --device "$DEVICE"
else
  mkdir -p "$OUT/preflight"
fi

python -m fate_x.engine.supervise_acpr_flowcal_foreground \
  --output_dir "$OUT/preflight" \
  --heartbeat_seconds 60 \
  --command python -m fate_x.engine.train_acpr_flowcal_pp --config "$CONFIG" --output_dir "$OUT/train" --device "$DEVICE" --epochs "$EPOCHS" --max_steps "$MAX_STEPS" --batch_size "$BATCH_SIZE" --beam_size "$BEAM_SIZE" --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" --checkpoint_every_steps "$CHECKPOINT_EVERY_STEPS"
