# ACPR FlowCal V2 Task Plan

Generated: 2026-06-23 00:12:56

## Current Objective

停止当前训练，固化 ACPR FlowCal V2 相对 ADAPT 复现的全过程记录、结果、问题和当前代码状态，并同步 GitHub branch。

## Source of Truth

- Remote repo: `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree`
- Branch: `flowtrace_pmt_v1`
- Current HEAD before this documentation update: `e34564b`
- Active V2 run inspected: `E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_staged_from_bestjoint_secaprotect_b8w6_20260622_191958\train`
- ADAPT repro base run: `E:\sbw\ADAPT_repro\ADAPT\output\repro_single_gpu\adapt_full_b4a16_20260522_011638`
- ADAPT repro resume run: `E:\sbw\ADAPT_repro\ADAPT\output\repro_single_gpu\adapt_full_b4a16_resume_latest_20260522_1530`

## Planned Contract From V2 Package

1. Start from the earlier strong ADAPT/FlowCal checkpoint, not from scratch.
2. Stage 1 `semantic_recovery`, 3 epochs:
   - Freeze Video Swin, control head, speed/course residual, BERT body.
   - Train reason target, PU/predicate utilities, reason memory, explanation SECA.
   - HardPair off.
3. Stage 2 `axis_aware_motion`, 5 epochs:
   - Enable longitudinal/lateral reason adapter, lane-wise flow statistics, speed/course residual.
   - Control reads reason memory with detach plus beta-gradient path.
   - BERT remains frozen.
4. Stage 3 `conflict_aware_joint`, 5 epochs:
   - Unfreeze BERT last layer and Video Swin last stage.
   - Apply gradient conflict handling on reason-related parameters.
   - HardPair low weight.
5. Stage 4 `flow_aware_scst`, 1-2 epochs:
   - Optimize explanation sequence reward only.
6. Stage 5 `sequence_calalign`, 1 epoch:
   - Fit alpha/temperature calibration parameters.

## Executed Run Status

- Training was stopped by user decision after V2 showed clear underperformance versus ADAPT reproduction.
- Latest completed V2 eval in the inspected run: epoch 4.
- Best V2 text metric so far: epoch 2 with `CIDEr_des+exp=2.0761`.
- The current V2 run did not pass the practical gate of beating ADAPT reproduction epoch 4.

## Remaining Status

- [x] Stop training and release GPU.
- [x] Extract ADAPT reproduction metrics.
- [x] Extract current V2 metrics.
- [x] Compare against ADAPT reproduction epoch 4 and best epoch.
- [x] Record observed implementation/runtime issues.
- [ ] Push current branch to GitHub after writing these records.

## Git Snapshot Before Documentation Commit

```text
M .gitignore
 M fate_x/engine/acpr_action_text_eval.py
 M fate_x/engine/acpr_bddx_data.py
 M fate_x/engine/acpr_control_eval.py
 M fate_x/engine/adapt_live_decoder_wrapper.py
 M fate_x/engine/audit_acpr_flowcal_pp.py
 M fate_x/engine/audit_flowtrace_pmt_implementation.py
 M fate_x/engine/backbone_output_utils.py
 M fate_x/engine/build_ablation_table.py
 M fate_x/engine/build_acpr_flow_atlas.py
 M fate_x/engine/build_flowtrace_atlas.py
 M fate_x/engine/build_reason_state_anchors.py
 M fate_x/engine/checkpoint_utils.py
 M fate_x/engine/eval_acpr_flowcal_pp.py
 M fate_x/engine/eval_flowtrace_pmt.py
 M fate_x/engine/eval_phrase_deletion.py
 M fate_x/engine/eval_phrase_faithfulness.py
 M fate_x/engine/export_acpr_flow_visuals.py
 M fate_x/engine/export_flowtrace_visuals.py
 M fate_x/engine/fate_x_compat.py
 M fate_x/engine/fit_sequence_calalign.py
 M fate_x/engine/flowtrace_adapt_bridge.py
 M fate_x/engine/generate_decoder_phrase_scores.py
 M fate_x/engine/generate_decoder_phrase_scores_from_model.py
 M fate_x/engine/generate_phrase_scores.py
 M fate_x/engine/lr_scaling.py
 M fate_x/engine/preflight.py
 M fate_x/engine/probe_acpr_flowcal_memory.py
 M fate_x/engine/probe_flowtrace_memory.py
 M fate_x/engine/run_acpr_flowcal_preflight_gates.py
 M fate_x/engine/smoke_fate_x_forward.py
 M fate_x/engine/supervise_acpr_flowcal_foreground.py
 M fate_x/engine/supervise_flowtrace_foreground.py
 M fate_x/engine/train_acpr_flowcal_pp.py
 M fate_x/engine/train_flowtrace_pmt.py
 M fate_x/engine/write_eval_artifacts.py
 M fate_x/losses/__init__.py
 M fate_x/losses/acpr_flowcal_losses.py
 M fate_x/losses/distillation.py
 M fate_x/losses/flowtrace_losses.py
 M fate_x/losses/segment_caption_loss.py
 M fate_x/losses/segment_caption_losses.py
 M src/layers/bert/modeling_bert.py
 M src/modeling/load_sensor_pred_head.py
?? .background_runs_F_full_junction_20260621_2240/
?? .codex/skills/acpr-flowcal-v2-implementation-audit/
?? REVIEW_PASS_ACPR_FLOWCAL_V2.txt
?? _check_linux_env.py
?? _cuda_test.py
?? _monitor_acpr_flowcalpp.py
?? acpr_eval_impl.patch
?? audit_v2_contract_remote.py
?? audit_v2_contract_remote_g.py
?? check_wsl_cuda.sh
?? configs/acpr_flowcal_v2_bddx_32f_224.yaml
?? control_temporal_fix_smoke.py
?? debug_v2_exit.sh
?? debug_v2_foreground.sh
?? docs/runbooks/ACPR_FlowCal_V2_File_Level_Checklist.md
?? docs/runbooks/ACPR_FlowCal_V2_Implementation_Manifest.json
?? docs/runbooks/ACPR_FlowCal_V2_Implementation_Plan.md
?? docs/runbooks/Codex_ACPR_FlowCal_V2_Bootstrap_Prompt.txt
?? docs/superpowers/supervision/2026-06-21-acpr-flowcal-v2.md
?? fate_x/acpr_flow_v2/
?? fate_x/engine/acpr_flowcal_v2_data.py
?? fate_x/engine/adapt_caption_eval_bridge.py
?? fate_x/engine/audit_acpr_flowcal_v2.py
?? fate_x/engine/build_acpr_flowcal_v2_atlas.py
?? fate_x/engine/eval_acpr_flowcal_v2.py
?? fate_x/engine/evaluate_v51_event_metrics.py
?? fate_x/engine/export_acpr_flowcal_v2_visuals.py
?? fate_x/engine/probe_acpr_flowcal_v2_memory.py
?? fate_x/engine/run_acpr_flowcal_v2_preflight.py
?? fate_x/engine/supervise_acpr_flowcal_v2_foreground.py
?? fate_x/engine/train_acpr_flowcal_v2.py
?? fate_x/explain/acpr_flowcal_v2_atlas.py
?? fate_x/explain/acpr_flowcal_v2_faithfulness.py
?? fate_x/explain/acpr_flowcal_v2_renderer.py
?? fate_x/losses/acpr_flowcal_v2_losses.py
?? fate_x/losses/explanation_scst.py
?? fix_control_state_temporal_code.py
?? fix_control_temporal_code.py
?? fix_v2_lineendings.py
?? inspect_pred_tsv.py
?? inspect_v2_files.sh
?? install_acpr_flowcal_v2_contract.py
?? install_v2_impl.py
?? install_v2_red_tests.py
?? launch_acpr_flowcal_v2_lf.ps1
?? launch_acpr_flowcal_v2_startprocess.ps1
?? launch_acpr_flowcal_v2_task.ps1
?? launch_acpr_flowcal_v2_wsl_full.sh
?? launch_acpr_v2_20260622_191505.ps1
?? launch_acpr_v2_20260622_191614.ps1
?? launch_acpr_v2_20260622_191958.ps1
?? launch_resume_controlfix.sh
?? monitor_acpr_flowcal_v2_wsl.sh
?? monitor_debug_v2.sh
?? patch_acpr_control_temporal.py
?? patch_v2_safe_logging.py
?? read_caption_head.py
?? read_label_head.py
?? run_acpr_v2_1904.sh
?? run_acpr_v2_20260622_191505.sh
?? run_acpr_v2_20260622_191614.sh
?? run_acpr_v2_20260622_191958.sh
?? run_acpr_v2_fast_auto_20260622.sh
?? run_linux_actionmetric_smoke.sh
?? run_linux_probe_b4w4.sh
?? run_linux_probe_b4w4_ld.sh
?? scan_car_info.py
?? scripts/FATE_X_acpr_flowcal_v2_foreground.ps1
?? scripts/FATE_X_acpr_flowcal_v2_foreground.sh
?? show_train_range.sh
?? stop_v2_debug.sh
?? test_v2_full_wsl.sh
?? test_v2_targeted_wsl.sh
?? tests/acpr_flowcal_v2/
?? tmp_run_acpr_v2_fast_auto_20260622.sh
?? tmp_search_traffic.sh
?? tmp_wait_fast_relaunch_20260622.ps1
?? wsl_persist_test.sh
```

## Diff Stat Before Documentation Commit

```text
.gitignore                                         |  323 +--
 fate_x/engine/acpr_action_text_eval.py             |  324 +--
 fate_x/engine/acpr_bddx_data.py                    |  300 +-
 fate_x/engine/acpr_control_eval.py                 |  300 +-
 fate_x/engine/adapt_live_decoder_wrapper.py        |  226 +-
 fate_x/engine/audit_acpr_flowcal_pp.py             |  526 ++--
 .../engine/audit_flowtrace_pmt_implementation.py   |  760 ++---
 fate_x/engine/backbone_output_utils.py             |   68 +-
 fate_x/engine/build_ablation_table.py              |  178 +-
 fate_x/engine/build_acpr_flow_atlas.py             |   32 +-
 fate_x/engine/build_flowtrace_atlas.py             |    6 +-
 fate_x/engine/build_reason_state_anchors.py        |  230 +-
 fate_x/engine/checkpoint_utils.py                  |   90 +-
 fate_x/engine/eval_acpr_flowcal_pp.py              |   52 +-
 fate_x/engine/eval_flowtrace_pmt.py                |  148 +-
 fate_x/engine/eval_phrase_deletion.py              |   12 +-
 fate_x/engine/eval_phrase_faithfulness.py          |  130 +-
 fate_x/engine/export_acpr_flow_visuals.py          |   46 +-
 fate_x/engine/export_flowtrace_visuals.py          |    6 +-
 fate_x/engine/fate_x_compat.py                     |   70 +-
 fate_x/engine/fit_sequence_calalign.py             |   44 +-
 fate_x/engine/flowtrace_adapt_bridge.py            |  464 +--
 fate_x/engine/generate_decoder_phrase_scores.py    |  386 +--
 .../generate_decoder_phrase_scores_from_model.py   |  294 +-
 fate_x/engine/generate_phrase_scores.py            |  206 +-
 fate_x/engine/lr_scaling.py                        |  168 +-
 fate_x/engine/preflight.py                         |   74 +-
 fate_x/engine/probe_acpr_flowcal_memory.py         |  314 +-
 fate_x/engine/probe_flowtrace_memory.py            |  156 +-
 fate_x/engine/run_acpr_flowcal_preflight_gates.py  |  314 +-
 fate_x/engine/smoke_fate_x_forward.py              |  244 +-
 fate_x/engine/supervise_acpr_flowcal_foreground.py |  124 +-
 fate_x/engine/supervise_flowtrace_foreground.py    |  172 +-
 fate_x/engine/train_acpr_flowcal_pp.py             | 2984 ++++++++++----------
 fate_x/engine/train_flowtrace_pmt.py               |  102 +-
 fate_x/engine/write_eval_artifacts.py              |  184 +-
 fate_x/losses/__init__.py                          |   10 +-
 fate_x/losses/acpr_flowcal_losses.py               |  154 +-
 fate_x/losses/distillation.py                      |   46 +-
 fate_x/losses/flowtrace_losses.py                  |  330 +--
 fate_x/losses/segment_caption_loss.py              |   24 +-
 fate_x/losses/segment_caption_losses.py            |   26 +-
 src/layers/bert/modeling_bert.py                   |   32 +-
 src/modeling/load_sensor_pred_head.py              |   50 +-
 44 files changed, 5379 insertions(+), 5350 deletions(-)
```
