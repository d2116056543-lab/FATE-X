# ACPR FlowCalPP / FlowCal V2 Unified Progress

Updated: 2026-06-23 00:25:13 Asia/Shanghai

## 1. 2026-05: ADAPT Reproduction And BDD-X Data Confirmation

- Remote ADAPT repo: `E:\sbw\ADAPT_repro\ADAPT`.
- User downloaded ADAPT processed BDD-X data.
- Confirmed `datasets` and `datasets_part` are both part of the BDD-X processed package.
- ADAPT caption/explanation evaluation uses COCO-style language metrics.
- BDD-X control is continuous speed/course, not discrete maintain/stop labels.
- Single-GPU reproduction was slower than the paper's multi-GPU setup.
- ADAPT reproduction completed through epoch 12.

ADAPT reproduction key results:

| Epoch | Checkpoint | CIDEr_des | CIDEr_exp | CIDEr_sum | speed RMSE | course RMSE |
|---:|---|---:|---:|---:|---:|---:|
| 4 | checkpoint-4-1024 | 2.3225 | 0.8410 | 3.1634 | 2.9790 | 6.1208 |
| 8 | checkpoint-8-2048 | 2.2545 | 0.9111 | 3.1656 | 2.4415 | 6.1165 |
| 12 | checkpoint-12-3072 | 2.3883 | 0.9107 | 3.2989 | 2.3625 | 6.1193 |

## 2. 2026-06-18: FlowTrace PMT V1

- Implemented initial FlowTrace / PMT mechanisms.
- Discovered smoke runs were insufficient because some max-step flags did not cap the actual loop.
- Lesson: a real smoke must verify actual `global_step` stop and emitted eval/checkpoint artifacts.

## 3. 2026-06-19 to 2026-06-21: ACPR FlowCalPP V1

Major implementation/evaluation items:

- Added ADAPT-aligned epoch-end test evaluation.
- Added best checkpoint logic.
- Added checkpoint-latest-before-eval behavior.
- Added continuous control evaluation and removed invalid discrete proxy as main metric.
- Added traffic-flow factors and audit logging.
- Added control prediction delta correlation fields.
- Ran WSL/Linux training because original ADAPT path is Linux-oriented.

Important runs and outcomes:

- `acpr_flowcalpp_adapt_testmetric_full_20260620_073402_fixedvis`
- `acpr_flowcalpp_adapt_testmetric_full_20260620_081725_hardpairfix`
- `acpr_linux_b4w4_resume_predcorrfix_20260621_034020`
- `acpr_linux_b4w4_resume_controltemporalfix_20260621_052010`
- `acpr_linux_b4w4_coursecircularfix3_20260621_121030`

Stopped V1 run:

- `E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030`
- stopped mid epoch 15 at `global_step=64500`, `optimizer_step=3717`.
- `checkpoint_latest.pth` valid.
- `checkpoint_latest.pth.tmp` invalid partial file.

V1 result summary:

- Strict/current epoch14: text poor, control close to ADAPT.
- Historical text best: text better than strict epoch14 but still below ADAPT reproduction.
- Traffic audit: factor-label relation visible; model-output causal usage not fully proven.

## 4. 2026-06-21 to 2026-06-22: ACPR FlowCal V2 Implementation

Implemented or added:

- `configs/acpr_flowcal_v2_bddx_32f_224.yaml`
- `fate_x/acpr_flow_v2/`
- `fate_x/engine/train_acpr_flowcal_v2.py`
- `fate_x/engine/eval_acpr_flowcal_v2.py`
- `fate_x/engine/adapt_caption_eval_bridge.py`
- `fate_x/losses/acpr_flowcal_v2_losses.py`
- `fate_x/losses/explanation_scst.py`
- `tests/acpr_flowcal_v2/`

Intended V2 stage flow:

1. semantic recovery
2. axis-aware motion
3. conflict-aware joint
4. flow-aware SCST
5. sequence calibration alignment

## 5. 2026-06-22: V2 Runs And Debugging

Key sequence:

1. Initial V2 run was slow and under-filled GPU.
2. User pointed out GPU had much more memory available.
3. Fast relaunch used `batch_size=32`, `num_workers=8`, `gradient_accumulation_steps=1`.
4. GPU memory rose to about 40GB.
5. Eval showed V2 still far below ADAPT reproduction and with broken course scale.
6. User requested stopping training and recording the outcome.

Main inspected V2 run:

- `E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191958\train`

V2 completed evals:

| Epoch | Stage | CIDEr_des | CIDEr_exp | CIDEr_sum | speed RMSE | course RMSE | pred speed corr | pred course corr |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | semantic_recovery | 1.3204 | 0.7272 | 2.0476 | 7.0222 | 88.9551 | -0.1775 | -0.1712 |
| 2 | semantic_recovery | 1.3208 | 0.7553 | 2.0761 | 7.0209 | 88.9550 | -0.1539 | -0.1470 |
| 4 | axis_aware_motion | 1.3210 | 0.7474 | 2.0685 | 6.9460 | 88.9634 | -0.1297 | -0.1221 |

## 6. 2026-06-23: Consolidation Fix

User correctly identified that separate V2-only markdown files split the experiment history.

Action taken:

- Folded V2 result and failure analysis into the existing ACPR FlowCalPP three-file ledger.
- Removed the separate `docs/runbooks/ACPR_FlowCal_V2_task_plan.md`, `docs/runbooks/ACPR_FlowCal_V2_findings.md`, and `docs/runbooks/ACPR_FlowCal_V2_progress.md` files from the canonical record set.
- Synchronized canonical files to local Downloads.
- Pushed branch after documentation correction.

## 7. Current Canonical Files

Remote:

- `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree\docs\acpr_flowcalpp\ACPR_FlowCalPP_task_plan.md`
- `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree\docs\acpr_flowcalpp\ACPR_FlowCalPP_findings.md`
- `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree\docs\acpr_flowcalpp\ACPR_FlowCalPP_progress.md`

Local:

- `C:\Users\WLJTXY\Downloads\ACPR_FlowCalPP_task_plan.md`
- `C:\Users\WLJTXY\Downloads\ACPR_FlowCalPP_findings.md`
- `C:\Users\WLJTXY\Downloads\ACPR_FlowCalPP_progress.md`

## 8. Current Stop State

- V2 training stopped.
- GPU released to desktop-level usage.
- No matching V2 train process remains.
- Latest pushed branch is `flowtrace_pmt_v1` on `d2116056543-lab/FATE-X`.
