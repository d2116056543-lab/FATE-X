# ACPR FlowCal V2 Progress Log

Generated: 2026-06-23 00:12:56

## Chronological Summary

1. Implemented ACPR FlowCal V2 code path around ADAPT/BDDX:
   - config: `configs/acpr_flowcal_v2_bddx_32f_224.yaml`
   - trainer: `fate_x/engine/train_acpr_flowcal_v2.py`
   - eval bridge: `fate_x/engine/eval_acpr_flowcal_v2.py`, `fate_x/engine/adapt_caption_eval_bridge.py`
   - V2 modules: `fate_x/acpr_flow_v2/`
   - V2 losses: `fate_x/losses/acpr_flowcal_v2_losses.py`, `fate_x/losses/explanation_scst.py`
   - tests: `tests/acpr_flowcal_v2/`

2. Initial V2 run was too slow and used low GPU memory.
   - Cause found: conservative batch/worker settings and staged freezing.
   - Fast relaunch used `batch_size=32`, `num_workers=8`, `gradient_accumulation_steps=1`.
   - GPU memory then reached about 40GB with high utilization.

3. Evaluation/checkpoint behavior was adjusted during the experiment:
   - Added ADAPT-style text evaluation outputs.
   - Added control metric outputs.
   - Added multiple best checkpoint files:
     - `checkpoint_best_text.pth`
     - `checkpoint_best_control.pth`
     - `checkpoint_best_adapt_joint.pth` / `checkpoint_best_joint.pth`
     - `checkpoint_best_test.pth`
   - Added traffic-flow audit fields including prediction-control correlation.

4. User requested stopping the run after results failed to beat ADAPT reproduction.
   - Stopped scheduled task `acpr_v2_fast_gpufill_20260622_2226`.
   - Confirmed no matching V2 Windows/WSL training process remained.
   - Confirmed GPU training memory was released.

## Current Run

- Run directory: `E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191958\train`
- Latest checkpoint observed: `E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191958\train\checkpoint_latest.pth`
- Latest completed eval directory: `E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191958\train\eval_epoch_004`
- Metrics file: `E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191958\train\metrics_summary.jsonl`

## V2 Eval Events

- Eval epoch 1 / stage `semantic_recovery`: CIDEr_des=1.3204, CIDEr_exp=0.7272, sum=2.0476, speed_rmse=7.0222, course_rmse=88.9551.
- Eval epoch 2 / stage `semantic_recovery`: CIDEr_des=1.3208, CIDEr_exp=0.7553, sum=2.0761, speed_rmse=7.0209, course_rmse=88.9550.
- Eval epoch 4 / stage `axis_aware_motion`: CIDEr_des=1.3210, CIDEr_exp=0.7474, sum=2.0685, speed_rmse=6.9460, course_rmse=88.9634.

## Last Train Progress Tail

- `{"epoch": 5, "batch": 240, "total_batches": 512, "stage": "axis_aware_motion", "loss": 3.0318169593811035}`
- `{"epoch": 5, "batch": 260, "total_batches": 512, "stage": "axis_aware_motion", "loss": 3.1186468601226807}`
- `{"epoch": 5, "batch": 280, "total_batches": 512, "stage": "axis_aware_motion", "loss": 3.019052028656006}`
- `{"epoch": 5, "batch": 300, "total_batches": 512, "stage": "axis_aware_motion", "loss": 3.1708219051361084}`
- `{"epoch": 5, "batch": 320, "total_batches": 512, "stage": "axis_aware_motion", "loss": 2.9564106464385986}`
- `{"epoch": 5, "batch": 340, "total_batches": 512, "stage": "axis_aware_motion", "loss": 3.2582383155822754}`
- `{"epoch": 5, "batch": 360, "total_batches": 512, "stage": "axis_aware_motion", "loss": 3.492183208465576}`
- `{"epoch": 5, "batch": 380, "total_batches": 512, "stage": "axis_aware_motion", "loss": 2.779273509979248}`
- `{"epoch": 5, "batch": 400, "total_batches": 512, "stage": "axis_aware_motion", "loss": 3.0346078872680664}`
- `{"epoch": 5, "batch": 420, "total_batches": 512, "stage": "axis_aware_motion", "loss": 2.915654420852661}`
- `{"epoch": 5, "batch": 440, "total_batches": 512, "stage": "axis_aware_motion", "loss": 3.1531386375427246}`
- `{"epoch": 5, "batch": 460, "total_batches": 512, "stage": "axis_aware_motion", "loss": 2.7259552478790283}`

## Problems Encountered And Actions

| Problem | Evidence | Action Taken | Status |
|---|---|---|---|
| Training was too slow / underfilled GPU | Earlier run used much less GPU memory than available | Relaunched with batch 32, workers 8, grad accum 1 | Runtime improved, metric quality not fixed |
| User wanted ADAPT-aligned metrics, not discrete action proxy | Discrete proxy collapsed and was not ADAPT legal metric | Removed discrete proxy from main reporting; kept ADAPT text/control metrics | Done for reporting |
| Traffic-flow audit lacked prediction-control relation | `pred_speed_delta_corr` / `pred_course_delta_corr` were null | Added/ran audit path to log prediction delta correlations | Now non-null but weak/negative |
| Best checkpoint updates appeared inconsistent after resume | Epoch 4 updated best files despite epoch 2 higher text sum | Identified likely selector history seeding issue | Needs code fix before next serious run |
| V2 control metric scale is far from ADAPT reproduction | V2 course RMSE about 89 vs ADAPT repro about 6.12 | Recorded as blocker | Unresolved |

## Git Context At Record Time

- Branch: `flowtrace_pmt_v1`
- HEAD before docs commit: `e34564b`
- Remote target: `github -> https://github.com/d2116056543-lab/FATE-X.git`

## Notes For Next Attempt

1. First run V2 eval with all V2 modules disabled on the exact ADAPT reproduction checkpoint.
2. Require metric parity with ADAPT reproduction before any training.
3. Fix control scale or target mismatch before motion stage.
4. Seed best-checkpoint selectors from historical metrics.
5. Only resume staged V2 if the no-op V2 bridge reproduces ADAPT checkpoint 4/12 metrics.
