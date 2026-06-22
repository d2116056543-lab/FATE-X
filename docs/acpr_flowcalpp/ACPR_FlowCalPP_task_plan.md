# ACPR FlowCalPP / FlowCal V2 Unified Task Plan

Updated: 2026-06-23 00:25:13 Asia/Shanghai

## 0. Recording Rule

This is the single task-plan ledger for the whole ACPR FlowCal line, from FlowTrace PMT / ACPR FlowCalPP V1 through ACPR FlowCal V2. Do not create a separate V2 task-plan file again. The three canonical files are:

- `docs/acpr_flowcalpp/ACPR_FlowCalPP_task_plan.md`
- `docs/acpr_flowcalpp/ACPR_FlowCalPP_findings.md`
- `docs/acpr_flowcalpp/ACPR_FlowCalPP_progress.md`

Local mirror:

- `C:\Users\WLJTXY\Downloads\ACPR_FlowCalPP_task_plan.md`
- `C:\Users\WLJTXY\Downloads\ACPR_FlowCalPP_findings.md`
- `C:\Users\WLJTXY\Downloads\ACPR_FlowCalPP_progress.md`

GitHub branch:

- `https://github.com/d2116056543-lab/FATE-X/tree/flowtrace_pmt_v1`

## 1. Global Objective

The task is not just to run ADAPT. The objective is to keep ADAPT-compatible BDD-X text/control evaluation while adding ACPR/FlowCal traffic-flow mechanisms that can explain or affect vehicle motion and generated explanations.

Hard boundaries:

- Use ADAPT/BDD-X legal text metrics for caption/explanation: BLEU, CIDEr, METEOR, ROUGE-L.
- Use ADAPT/BDD-X continuous control metrics for speed/course: RMSE and threshold accuracies.
- Do not use the extra discrete maintain/stop/straight/turn proxy as a main metric.
- Do not claim traffic-flow causality from correlation alone; require model-output audit or intervention evidence.
- Do not continue a long run if the no-op bridge cannot reproduce ADAPT-like checkpoint behavior.

## 2. Data And Baseline Scope

Primary reference repo and data:

- ADAPT repo: `E:\sbw\ADAPT_repro\ADAPT`
- BDD-X processed data: `E:\sbw\ADAPT_repro\ADAPT\ADAPT_PREPROCESSED_DATASET`
- ADAPT reproduction base run: `E:\sbw\ADAPT_repro\ADAPT\output\repro_single_gpu\adapt_full_b4a16_20260522_011638`
- ADAPT reproduction resume run: `E:\sbw\ADAPT_repro\ADAPT\output\repro_single_gpu\adapt_full_b4a16_resume_latest_20260522_1530`

ADAPT reproduction checkpoints used as comparison:

- Epoch 4: `checkpoint-4-1024`, `CIDEr_des+exp=3.1634`, speed RMSE `2.9790`, course RMSE `6.1208`.
- Best observed epoch 12: `checkpoint-12-3072`, `CIDEr_des+exp=3.2989`, speed RMSE `2.3625`, course RMSE `6.1193`.

## 3. V1 / FlowCalPP Plan

V1 was the first ACPR FlowCalPP line. Its purpose was to align with ADAPT eval while adding traffic-flow / predicate / reason-control mechanisms.

Core requirements:

1. Add real epoch-end ADAPT-style test evaluation.
2. Save `checkpoint_latest.pth` before evaluation so eval failures do not lose training progress.
3. Save separate best checkpoints:
   - `checkpoint_best_text.pth`
   - `checkpoint_best_control.pth`
   - `checkpoint_best_adapt_joint.pth` or equivalent joint selector
4. Keep text metrics and continuous control metrics separate.
5. Log traffic-flow audits, including predicted control delta correlations.
6. Avoid using discrete action proxy as main selection metric.

V1 current stopped run:

- Run: `E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030`
- Stop point: epoch 15 mid-run, `global_step=64500`, `optimizer_step=3717`.
- Effective resume checkpoint: `train\checkpoint_latest.pth`.
- Invalid partial file: `train\checkpoint_latest.pth.tmp`; do not resume from `.tmp`.
- This stopped point did not complete a new test eval.

## 4. V1 Result Gate

V1 produced useful engineering fixes but did not beat ADAPT text. It showed control was near ADAPT, while text remained behind.

Strict/current mechanism epoch14:

- `CIDEr_des=0.6488`
- `CIDEr_exp=0.5441`
- `CIDEr_des+exp=1.1930`
- speed RMSE `2.5398`
- course RMSE `6.1176`

Historical text best from earlier diagnostics:

- `CIDEr_des=1.4878`
- `CIDEr_exp=0.7872`
- `CIDEr_des+exp=2.2750`
- speed RMSE `2.6372`
- course RMSE `6.1163`

Interpretation:

- V1 control reached ADAPT-like range.
- V1 text did not approach ADAPT reproduction epoch4 or paper-level text quality.
- Traffic-flow fields became logged, but needed stronger intervention/zero-out proof.

## 5. V2 Plan

V2 was intended to start from the stronger early checkpoint and add FlowCal mechanisms in stages instead of training everything from scratch.

Planned staged schedule:

1. `semantic_recovery`, 3 epochs:
   - Freeze Video Swin, control head, speed/course residual, BERT body.
   - Train reason target, PU/predicate utilities, reason memory, explanation SECA.
   - HardPair off.
2. `axis_aware_motion`, 5 epochs:
   - Enable longitudinal/lateral reason adapter, lane-wise flow statistics, speed/course residual.
   - Control reads reason memory with detach plus beta-gradient path.
   - BERT remains frozen.
3. `conflict_aware_joint`, 5 epochs:
   - Unfreeze BERT last layer and Video Swin last stage.
   - Apply gradient-conflict handling on reason-related parameters.
   - Start HardPair at low weight.
4. `flow_aware_scst`, 1-2 epochs:
   - Optimize explanation sequence reward.
5. `sequence_calalign`, 1 epoch:
   - Fit alpha/temperature calibration parameters.

V2 implementation locations:

- Config: `configs/acpr_flowcal_v2_bddx_32f_224.yaml`
- Modules: `fate_x/acpr_flow_v2/`
- Trainer: `fate_x/engine/train_acpr_flowcal_v2.py`
- Eval: `fate_x/engine/eval_acpr_flowcal_v2.py`
- ADAPT caption eval bridge: `fate_x/engine/adapt_caption_eval_bridge.py`
- Tests: `tests/acpr_flowcal_v2/`

## 6. V2 Executed Run

Main inspected run:

- `E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191958\train`

Fast relaunch configuration after speed issue:

- `batch_size=32`
- `num_workers=8`
- `gradient_accumulation_steps=1`
- GPU memory reached about 40GB, but metrics still did not improve enough.

Latest completed V2 eval:

- Completed through epoch 4.
- Best V2 by text sum was epoch 2.
- Training was stopped because it was clearly below ADAPT reproduction.

## 7. Final Decision Before Next Run

Do not continue V2 training as-is.

Before any new full run, required gates are:

1. Run pure V2 evaluation on the intended resume checkpoint with V2 mechanisms disabled.
2. Prove this no-op V2 bridge reproduces ADAPT checkpoint metrics.
3. Fix or explain the course control scale mismatch.
4. Seed best-checkpoint selectors from full historical `metrics_summary.jsonl`, not only the current latest checkpoint payload.
5. Only then restart staged V2 training.

## 8. Active Next Task

Current task is documentation consolidation, not training:

- [x] Stop V2 training.
- [x] Merge V1 and V2 records into the same three canonical ACPR FlowCalPP markdown files.
- [x] Remove the separate V2-only task/findings/progress files from the canonical record set.
- [ ] Sync canonical files to local Downloads.
- [ ] Commit and push the updated branch.
