# FATE-X Implementation Status 2026-05-25

## Required Context

Before starting any remote FATE-X task, read:

- `E:\sbw\FATE_Drive\task_plan.md`
- `E:\sbw\FATE_Drive\findings.md`
- `E:\sbw\FATE_Drive\progress.md`

## What Is Implemented

FATE-X now contains ADAPT-compatible token reduction, temporal evidence memory, hook smoke, and phrase score generation/evaluation.

Implemented modules:

- Token reducer: `fate_x/models/video_token_reducer.py`
- Temporal evidence memory: `fate_x/models/temporal_evidence_memory.py`
- Phrase counterfactual utilities: `fate_x/explain/phrase_counterfactual.py`
- Video-token attribution helpers: `fate_x/explain/video_token_attribution.py`
- Phrase deletion/sufficiency evaluator: `fate_x/engine/eval_phrase_faithfulness.py`, `fate_x/engine/eval_phrase_deletion.py`
- Phrase score generation from predictions + optional token scores: `fate_x/engine/generate_phrase_scores.py`
- ADAPT hook smoke: `fate_x/engine/smoke_fate_x_forward.py`

ADAPT integration:

- `src/modeling/multitask_e2e_vid_swin_bert.py`
- `src/modeling/video_captioning_e2e_vid_swin_bert.py`

The FATE-X path is default-off and preserves original ADAPT behavior unless enabled with flags.

## Hook Forward Smoke

Smoke command:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_x.engine.smoke_fate_x_forward ^
  --output .background_runs\fate_x_hook_smoke_20260525.json ^
  --video_token_reducer topk_merge ^
  --temporal_evidence_memory queries ^
  --batch_size 2 --num_tokens 64 --max_img_seq_length 64 --dim 64
```

Verified output:

```json
{
  "input_shape": [2, 64, 64],
  "output_shape": [2, 48, 64],
  "attention_mask_shape": [2, 64, 64],
  "token_stats": {
    "original_tokens": 64,
    "kept_tokens": 32,
    "summary_tokens": 8,
    "reduced_tokens": 40,
    "event_tokens": 8
  },
  "has_provenance": true
}
```

Interpretation:

- reducer compressed 64 video tokens to 40 reduced tokens
- temporal evidence memory added 8 event tokens
- final visual sequence length is 48
- multimodal attention mask resized to text length plus visual length
- provenance exists for reduced tokens

## Phrase Score Generation

`generate_phrase_scores.py` converts prediction JSONL plus optional token scores into phrase-score JSONL consumable by `eval_phrase_deletion.py`.

Smoke result:

```text
count 1
with_phrase_hit 1
with_generated_scores 1
faithfulness_available true
phrase_deletion_score 0.05
phrase_sufficiency_score 0.80
random_deletion_score 0.35
```

Important boundary:

- This proves the score-generation/evaluation pipeline works.
- Full generation-time faithfulness still requires running a trained ADAPT/FATE-X checkpoint to produce real phrase log-prob or token attribution scores.

## Verification

Fresh GitHub zip compile:

- `COMPILED_FILES 46`
- `FRESH_ZIP_PY_COMPILE_OK`

Remote tests:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest ^
  tests/test_fate_x_modules.py ^
  tests/test_fate_x_phrase_counterfactual.py ^
  tests/test_fate_x_video_token_attribution.py ^
  tests/test_fate_x_forward_and_phrase_scores.py -q
```

Result:

```text
8 passed
```

## What Is Still Not A Final Result

The FATE-X method code path is implemented and smoke-tested, but final paper-style results still require:

- full ADAPT/FATE-X training or inference with a trained checkpoint
- caption/control metrics under baseline ADAPT and FATE-X flags
- real generation-time phrase log-prob extraction
- final phrase deletion/sufficiency table on val/test

## 2026-05-25 Gap Closure Update

The latest GPTPro audit identified two remaining FATE-X code-level gaps:

- unsafe coexistence of ADAPT `learn_mask_enabled` and compressed FATE-X video tokens;
- phrase faithfulness needed a decoder-token-logprob scoring entrypoint, not only token-score fallback fields.

Both are now implemented and verified.

### New/Updated Code

- `fate_x/engine/fate_x_compat.py`
  - Adds `validate_fate_x_mask_compatibility(args)`.
  - Raises `ValueError` when `fate_x_enabled=True`, `video_token_reducer != none`, and `learn_mask_enabled=True`.

- `src/tasks/run_adapt.py`
  - Calls the compatibility guard in argument checking before training starts.

- `fate_x/engine/generate_decoder_phrase_scores.py`
  - Reads generated text, decoder tokens, and token log-probs.
  - Supports optional top-k masked, evidence-only, and random-masked token log-prob fields.
  - Produces phrase-level `original_score`, `topk_masked_score`, `evidence_only_score`, and `random_masked_score`.
  - Summarizes phrase deletion, sufficiency, and random deletion scores.

- `tests/test_fate_x_forward_and_phrase_scores.py`
  - Adds compatibility-guard test.
  - Adds decoder phrase scoring test.

### Verification

Targeted tests:

```text
tests/test_fate_x_forward_and_phrase_scores.py: 4 passed
```

Broader selected tests:

```text
10 passed
```

Real script smoke:

```text
decoder phrase scorer:
  count = 1
  with_phrase_hit = 1
  phrase_records = 3
  faithfulness_available = true
  phrase_deletion_score = 0.80
  phrase_sufficiency_score = -0.1333
  random_deletion_score = 0.2167

forward hook smoke:
  input_shape = [1, 24, 16]
  output_shape = [1, 16, 16]
  attention_mask_shape = [1, 20, 20]
  original_tokens = 24
  reduced_tokens = 14
  event_tokens = 2
  has_provenance = true
```

Fresh GitHub clone compile:

```text
FATE-X commit 242bd0447c34eab2cbbe24ba49dd6e652d62981b
py_compile PASS
```

Boundary remains unchanged: full checkpoint-dependent ADAPT/FATE-X phrase faithfulness still requires real decoder inference outputs from a trained checkpoint.
