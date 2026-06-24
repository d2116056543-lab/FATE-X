<!--
Canonical ACPR FlowCalPP / FlowCal V2 Task Plan ledger.
This file intentionally contains both the earlier V1/FlowCalPP record and the later V2 record.
Restored and merged from git commit ccb0370 on 2026-06-23 00:32:59 Asia/Shanghai after the user requested one continuous three-file history.
Do not split V2 into separate task/findings/progress files again.
-->

# Unified ACPR FlowCalPP / FlowCal V2 Task Plan Ledger

Updated: 2026-06-23 00:32:59 Asia/Shanghai

## Record Policy

This is one of the three canonical records for the whole ACPR line, from ADAPT reproduction through FlowTrace PMT, ACPR FlowCalPP V1, and ACPR FlowCal V2. The goal is to preserve enough detail to avoid repeating failed training, evaluation, checkpoint, and metric-alignment mistakes.

Canonical files:

- `docs/acpr_flowcalpp/ACPR_FlowCalPP_task_plan.md`
- `docs/acpr_flowcalpp/ACPR_FlowCalPP_findings.md`
- `docs/acpr_flowcalpp/ACPR_FlowCalPP_progress.md`

Local mirrors:

- `C:\Users\WLJTXY\Downloads\ACPR_FlowCalPP_task_plan.md`
- `C:\Users\WLJTXY\Downloads\ACPR_FlowCalPP_findings.md`
- `C:\Users\WLJTXY\Downloads\ACPR_FlowCalPP_progress.md`

The first section below is the detailed V1/FlowCalPP record. The second section appends the V2-specific record. Both are kept because the V2 failure only makes sense in the context of the V1 ADAPT-aligned eval and resume history.

---

# Part A: Detailed V1 / FlowCalPP Record

# ACPR FlowCalPP Task Plan

更新时间：2026-06-21 14:25 Asia/Shanghai

## 0. 记录规范

后续只维护这三份 ACPR/FlowCalPP 专用记录，不再把关键信息散落到聊天、临时脚本或额外 md：

- `ACPR_FlowCalPP_task_plan.md`：任务范围、实验协议、当前状态、下一步计划、防复发规则。
- `ACPR_FlowCalPP_findings.md`：结果、组件作用、ADAPT 对比、问题根因、错误经验库。
- `ACPR_FlowCalPP_progress.md`：时间顺序日志、代码修改链、运行链、验证链、GitHub 同步状态。

同步位置：

- 远端仓库：`E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree\docs\acpr_flowcalpp\`
- 本地记录：`E:\FATE_X_ACPR_FlowCalPP_Records\`
- GitHub branch：`d2116056543-lab/FATE-X` 的 `flowtrace_pmt_v1`

记录要求：

- 每次训练、暂停、修复、评估、失败都必须写入这三份文件之一。
- 每个失败都要包含：触发条件、现象、根因、修复、验证、防复发结论。
- 不能只写“已修复/已完成”，必须写验证证据和仍然不能下的结论。
- 不能混用 ADAPT 合法指标和后来添加的诊断 proxy；指标口径必须写清楚。

## 1. 当前状态

- 远端代码目录：`E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree`
- 当前 branch：`flowtrace_pmt_v1`
- 当前训练已按用户要求停止。
- 停止前任务名：`acpr_coursefix3_121030`
- 停止前 run：`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030`
- 停止点：epoch `15` 中途，`global_step=64500`，`optimizer_step=3717`。
- 最后 batch loss：`0.7540768981`。
- 有效续训点：`...\train\checkpoint_latest.pth`，大小 `2785647147` bytes，写入时间 `2026-06-21 12:46:50`。
- 无效临时文件：`...\train\checkpoint_latest.pth.tmp`，大小 `1291058816` bytes，写入时间 `2026-06-21 13:00:53`。这是停止时残留的 partial tmp，不能用于 resume。
- 该 stopped run 没有完成新的 test eval，不能拿它和 ADAPT 或历史 best 做结果对比。

## 2. 实验目标边界

本任务不是纯 ADAPT 复现，也不是只看 caption。目标是：

- 数据、test split、文本评价指标、连续 control 评价指标尽量与 ADAPT 口径对齐。
- 在 ADAPT 合法评价基础上加入 ACPR/FlowCalPP 的交通流、predicate、reason/control、hardpair、future-control 等机制。
- 不把 speed/course 强行转成 maintain/stop/straight/turn 作为主评价，因为 BDD-X 原始提供的是连续 control signal。
- 保留文本生成评价，因为 ADAPT 主任务就是 description/explanation 生成。
- 保留连续 speed/course control 评价，因为用户任务需要车辆运动控制结果。
- 交通流因子必须有审计输出，不能只作为训练时不可见的黑盒模块。

## 3. 固定实验协议

数据：

- 使用 BDD-X 32-frame processed 数据。
- 训练图像和文本沿用 ADAPT processed dataset 结构。
- test split 使用 ADAPT 的 `datasets/BDDX/testing_32frames.yaml`。
- 连续控制信号使用 `datasets_part/BDDX/testing_32frames.yaml`。
- 不使用额外手工离散化标签作为主评价。

文本评价：

- 使用 ADAPT/COCO caption evaluation 口径。
- 分别评估 `description` 和 `explanation`。
- 主文本选择指标：`CIDEr_des + CIDEr_exp`。

Control 评价：

- 使用连续 speed/course 口径。
- 输出 `RMSE`、`MAE`、`Acc@0.1`、`Acc@0.5`、`Acc@1`、`Acc@5`、`Acc@10`。
- `checkpoint_best_control.pth` 不能由离散 proxy 决定，只能由连续 control 合法指标决定。

Checkpoint：

- `checkpoint_latest.pth`：每轮 eval 前先保存，用于防止 eval 崩溃导致该轮训练白跑。
- `checkpoint_best_text.pth`：按 `CIDEr_des + CIDEr_exp`。
- `checkpoint_best_control.pth`：按 continuous control 合法选择分数。
- `checkpoint_best_adapt_joint.pth`：按 ADAPT 合法文本 + continuous control 综合分数。
- `checkpoint_best_test.pth`：按当前配置的 test selection metric。
- 不允许用 `.tmp` ckpt resume。

## 4. 已落地的主功能

- ADAPT 对齐的 per-epoch test eval。
- ADAPT/COCO caption metrics：description/explanation 分开输出。
- continuous speed/course control metrics。
- eval 前保存 `checkpoint_latest.pth`。
- best_text / best_control / best_adapt_joint / best_test 分开保存。
- 删除或降级旧的离散 decision proxy，不作为主指标。
- traffic-flow audit 增加 target control delta 和 predicted control delta 的相关性。
- course delta circular fix 已写入代码，但 stopped run 尚未完成新的 eval 验证。
- hardpair formal-path 梯度问题已修复并通过测试。
- 文档和代码已推送到 `flowtrace_pmt_v1` branch。

## 5. 当前有效结果分层

### 5.1 Strict/current mechanism result

有效文件：

`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_resume_envfix_20260621_065540\train\eval_epoch_014.json`

关键结论：

- `CIDEr_des+exp = 1.1929573563`
- `CIDEr_des = 0.6488124820`
- `CIDEr_exp = 0.5441448743`
- `speed RMSE = 2.5398054123`
- `speed Acc@0.5 = 0.2893738151`
- `speed Acc@1 = 0.4492167234`
- `course RMSE = 6.1176033020`
- `course Acc@0.5 = 0.8675665259`
- `course Acc@1 = 0.9101654291`
- traffic audit 已经包含 `pred_speed_delta_corr` 和 `pred_course_delta_corr`。

注意：这是 course circular fix 前的结果。

### 5.2 Historical text best

有效文件：

`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_resume_epoch5_foreground_20260620_185230\train\eval_epoch_005.json`

关键结论：

- `CIDEr_des+exp = 2.2750277139`
- `CIDEr_des = 1.4878288790`
- `CIDEr_exp = 0.7871988349`
- `speed RMSE = 2.6372163296`
- `course RMSE = 6.1162762642`

注意：该点文本明显更强，但属于旧诊断口径时期，不代表当前 strict mechanism 的唯一结论。

## 6. 与 ADAPT 原文差距

ADAPT 原文表格约等价小数口径：

- `CIDEr_des ≈ 2.475`
- `CIDEr_exp ≈ 1.026`
- `CIDEr_des+exp ≈ 3.501`
- `speed RMSE ≈ 2.5`
- `speed Acc@0.5 ≈ 0.281`
- `speed Acc@1 ≈ 0.453`
- `course RMSE ≈ 6.4`
- `course Acc@0.5 ≈ 0.855`
- `course Acc@1 ≈ 0.899`

当前判断：

- continuous control 已接近或部分略优于 ADAPT 表格口径。
- text generation 仍显著落后，strict epoch14 只有 ADAPT 文本 CIDEr 合计约 34%，historical text best 约 65%。
- 如果目标是“和 ADAPT 文本结果接近”，下一步不能只继续堆 traffic/control 分支，要回查 caption loss、beam/max_len、tokenizer、label 对齐和文本分支权重。

## 7. 不再重复的错误

- 不再把“能跑训练”当作实现完成。必须有 per-epoch test eval、best ckpt、resume、metrics JSON。
- 不再把 continuous control 强行离散化后当主评价。
- 不再在 eval 后才保存 latest；必须 eval 前保存。
- 不再只看 CIDEr 而忽略 control。
- 不再只看 target/control 相关性；必须同时输出模型预测 control delta 的相关性。
- 不再用 SSH 超长嵌套 PowerShell one-liner 做关键验证；优先临时脚本或 EncodedCommand。
- 不再在有 `.tmp` ckpt 时直接续训；必须确认有效 `checkpoint_latest.pth`。
- 不再在 hardpair raw loss 有数值时假设 projection 有梯度；必须有 formal-path 测试验证。
- 不再把 historical text best 和 strict current mechanism best 混成同一个结论。

## 8. 下一步计划

优先级 1：从有效 `checkpoint_latest.pth` 续训至少跑完一个完整 epoch，产出 course circular fix 后的新 eval。

优先级 2：如果新 eval 的 text CIDEr 继续低于 historical text best，暂停增强 control/traffic loss，集中审查：

- caption action/explanation loss 权重是否被其他辅助 loss 稀释。
- beam size / max length 是否与 ADAPT 原文或复现配置一致。
- tokenizer 和 `description/explanation` label 是否完全对齐。
- teacher forcing、caption head、BERT/Transformer 初始化是否被改动。
- ACPR/FlowCalPP 分支是否通过 shared representation 抑制了 caption 表达能力。

优先级 3：traffic-flow 机制做 zero-out / counterfactual audit：

- zero queue_congestion / clear_open_flow / traffic_signal / lead_vehicle_group。
- 比较 zero-out 前后 `speed/course RMSE`、`CIDEr`、predicted control delta。
- 只有 zero-out 证明模型输出依赖这些因子后，才能说交通流机制不只是相关性。

优先级 4：继续保留三类 checkpoint，不合并 best 文件。

## 9. 后续运行前检查清单

- 训练进程是否真的停止或已启动：检查 `Get-CimInstance Win32_Process`。
- 远端 cwd 是否为 `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree`。
- WSL/Linux 环境是否使用 `/opt/conda/envs/adapt/bin/python`。
- 是否使用 `checkpoint_latest.pth` 而不是 `.tmp`。
- 是否 eval 前保存 latest。
- 是否每轮产出 `eval_epoch_XXX.json`。
- 是否同时有 text/control/joint best。
- 是否 traffic audit 中 `pred_speed_delta_corr` 和 `pred_course_delta_corr` 不为 null。
- 是否没有恢复旧离散 proxy 作为主指标。


---

# Part B: Detailed V2 Record Appended To The Same Ledger

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

## ACPR-DynFlow V1 strict plan continuation (2026-06-23 03:45 China time)

### Contract source
- Package: `C:/Users/WLJTXY/Downloads/FATE_X_ACPR_DynFlow_V1_Codex_Package_20260622`.
- Branch/worktree required by plan: `acpr_dynflow_v1` at `E:/sbw/FATE_Drive/fate_x_acpr_dynflow_v1_worktree`.
- Base branch: `flowtrace_pmt_v1`, clean source HEAD `8a52f4e99e81406cd949afaadb16c7483cf5025d`.
- Formal target namespace: `fate_x/acpr_dynflow`.

### Non-negotiable plan gates
- Direct-image BDD-X input must remain `[B, 32, 3, 224, 224]`; no cached ADAPT features, no cached logits, no token cache shortcut.
- Formal training must be an independent model path. It must not resume ADAPT, FlowTrace, FlowCalPP, or V2 checkpoints.
- Allowed initializers only: generic BERT base, Kinetics Video Swin, and the user's BDD-OIA ACPR-CalAlign predicate-query checkpoint.
- The 32 predicate ontology must match `fate_oia_acpr_calalign_v1_2` exactly.
- The official formal trainer must be `fate_x.engine.train_acpr_dynflow`; evaluator must be `fate_x.engine.eval_acpr_dynflow`.
- Preflight reports A-L and `REVIEW_PASS_ACPR_DYNFLOW_V1.txt` are required before formal training.
- Plan explicitly requires foreground-supervised formal training and forbids detached formal training before review pass.

### Implemented code-level surface
- Added config and runbook artifacts: `configs/acpr_dynflow_v1_bddx_32f_224.yaml`, `configs/acpr_dynflow_predicates.yaml`, `configs/acpr_dynflow_text_rules.yaml`, `configs/acpr_dynflow_traffic_grammar.yaml`, `docs/runbooks/ACPR_DynFlow_V1_Implementation_Plan.md`, `docs/runbooks/ACPR_DynFlow_V1_Implementation_Manifest.json`, `docs/runbooks/ACPR_DynFlow_V1_File_Level_Checklist.md`, `.codex/skills/acpr-dynflow-implementation-audit/SKILL.md`.
- Added `fate_x/acpr_dynflow` modules for signal codec, predicate ontology/transfer, ego motion, dynamic predicate field, NNPU/CalAlign, homogenizer, pattern router, lane flow, traffic state reasoner, response lag, global decision stream, decision ledger, contribution alignment, text decoder, interventions, and integrated model.
- Added official engine entrypoints: `fate_x.engine.train_acpr_dynflow`, `fate_x.engine.eval_acpr_dynflow`, `fate_x.engine.run_acpr_dynflow_preflight`, `fate_x.engine.audit_acpr_dynflow`, `fate_x.engine.probe_acpr_dynflow_memory`, `fate_x.engine.export_acpr_dynflow_visuals`, `fate_x.engine.build_acpr_dynflow_atlas`, `fate_x.engine.supervise_acpr_dynflow_foreground`.
- Added formal explanation/visualization support in `fate_x/explain/acpr_dynflow_*`.
- Added `tests/acpr_dynflow` coverage for model shapes, loss, data, preflight artifacts, intervention hooks, factor identity, and engine entrypoints.

### Current execution policy
- Synthetic smoke is allowed for code-path validation.
- Formal BDD-X training is not allowed until the OIA ACPR-CalAlign predicate-query checkpoint is resolved and review pass is clean.
- If the user wants to override this gate, record it explicitly as a protocol deviation; do not silently call it plan-compliant.

## ACPR-DynFlow V1 real-loader hardening (2026-06-23 04:16 China time)

### Why this was required
- The first implementation had functional direct-image tensor flow, but `video_backbone.py` used a lightweight Conv/GRU fallback and `text_decoder.py` used a lightweight embedding path.
- That was not sufficient for the plan clause requiring Kinetics Video Swin and generic BERT-base as the only formal initializers.

### New implementation commitment
- `fate_x/acpr_dynflow/video_backbone.py` now instantiates `torchvision.models.video.swin3d_b`.
- The official Video Swin K600 checkpoint is downloaded to `models/video_swin_transformer/swin_base_patch244_window877_kinetics600_22k.pth`.
- The official checkpoint keyspace is converted deterministically from `backbone.layers.*` / `mlp.fc1/fc2` to torchvision `features.*` / `mlp.0/3`.
- The loader converted `351/353` keys; the only skipped/missing tensors are the classifier head (`cls_head.fc_cls.*` vs torchvision `head.*`), which is expected because DynFlow uses feature states, not the pretrained classifier head.
- `fate_x/acpr_dynflow/text_decoder.py` now loads local `models/captioning/bert-base-uncased` with `transformers.BertModel`, freezes embeddings and bottom 8 encoder layers, and trains the top 4 layers.
- `fate_x/engine/run_acpr_dynflow_preflight.py` now writes `backbone_audit.json` and `text_decoder_audit.json` with explicit `kinetics_loaded`, `uses_torchvision_swin`, and `bert_loaded` fields.
- `fate_x/engine/audit_acpr_dynflow.py` now blocks formal review if Video Swin or BERT assets are missing or fail to load.

## 2026-06-23 ACPR-DynFlow V1 OIA/Gradient Gate 补全计划更新

### 当前必须完成的剩余门槛
- 已完成：将 `paths.oia_acpr_checkpoint` 从 unresolved placeholder 改为真实 CalAlign/OIA predicate checkpoint：`E:/sbw/FATE_Drive/fate_oia_acpr_calalign_v1_2_worktree/.background_runs/acpr_calalign_v1_2_resume_e15_17_sched28_20260616_125105/checkpoint_best_test_final_calibrated.pth`。
- 已完成：`PredicateQueryInitializer` 不再只用随机 prior/name prior；现在真实读取 checkpoint 内 `model/predicate_head.predicate_queries`，记录 SHA256、source key、source dim、extra predicate-head keys。
- 已完成：`run_acpr_dynflow_preflight.py` 写出 `state["oia_loaded"]`，`audit_acpr_dynflow.py` 在 `oia_loaded=false` 时阻断为 `oia_query_not_loaded`。
- 已完成：`maybe_write_pass` 不再只看 missing reports；现在会读取 required audit JSON 中的 `passed:false` 并用 `failed_required_reports` 阻断，避免 gate false 但 review pass 误通过。
- 已完成：`gate_gradient_chain` 从错误的“每个 trainable parameter 都必须非零”修正为计划要求的“每个 intended trainable component 有有限非零梯度，冻结组件无梯度”，并保留 parameter-level zero/missing 列表用于诊断。
- 已完成：修复真实梯度断点，而不是只放宽 gate：`lane_flow.encoder` 进入 `TrafficStateReasoner`，`lateral_bias` 进入 `DecisionLedgerHead` 的 course contribution，`bb.text_visual_tokens` 进入 `DynFlowTextDecoder`，未使用的 `swin.head` 和 `bert.pooler` 冻结。

### 下一步仍必须执行
- 提交并 push 当前 9 个文件修改到 GitHub `acpr_dynflow_v1`。
- 在 clean HEAD 上重新运行动态 preflight，要求 `review_report.blockers=[]` 且 `failed_reports=[]`。
- 运行正式 audit 并写 `REVIEW_PASS_ACPR_DYNFLOW_V1.txt`，否则不得启动 formal training。

## 2026-06-23 04:52 Final Preflight / Review Pass Evidence

- Clean HEAD at time of pass: `9f8c122b2ad36f704591c4f7c0a7796cb4ac825d`.
- Dynamic preflight output dir: `.background_runs/acpr_dynflow_v1_final_preflight_20260623_0450`.
- `review_report.json`: `passed=true`, `blockers=[]`, `missing_reports=[]`, `failed_reports=[]`.
- Formal audit with `--write_review_pass`: exit 0 and wrote `REVIEW_PASS_ACPR_DYNFLOW_V1.txt`.
- OIA proof: `oia_loaded=true`, `oia_source=model`, `oia_source_dim=384`, `oia_prior_shape=32x384`, checkpoint SHA `84d3744a7505cca19b33ac2b517b58d71c98fd580f162dec4a6eee2aee1f64b2`.
- Gradient proof: `gate_gradient_chain.passed=true`, `missing_components=[]`, `frozen_params_with_grad=[]`, `missing_trainable_grad_params=0`.
- Git proof: local HEAD equals GitHub `acpr_dynflow_v1` HEAD at `9f8c122b2ad36f704591c4f7c0a7796cb4ac825d`.
- Note: This md append changes HEAD, so final authorization must be rerun after this documentation commit.

---

## Unified ACPR/ADAPT Timeline Expansion - 2026-06-23 05:09:06 +08:00

### Why this expansion exists
The previous three-record set was too compressed for the amount of work done across ADAPT reproduction, FlowTrace/FlowCalPP, V2, and DynFlow. This section turns the records into the durable source of truth for future runs: what was tried, which metrics moved, which implementation details were corrected, and which mistakes must not be repeated.

### Scope covered by this unified record
- ADAPT reproduction on BDD-X preprocessed 32-frame TSV data.
- FlowTrace PMT V1 and ACPR FlowCalPP V1/V2 experiments layered on top of the ADAPT-style data path.
- DynFlow V1 strict implementation package under
ate_x/acpr_dynflow.
- Training/evaluation contract decisions: ADAPT-style caption metrics, continuous control metrics, checkpoint selection, and forbidden discrete proxy evaluation.
- Git synchronization and review-pass rules for d2116056543-lab/FATE-X branches.

### Chronological task map
| Phase | Main branch/worktree | Intended goal | Actual status | Durable lesson |
|---|---|---|---|---|
| ADAPT reproduction | E:/sbw/ADAPT_repro/ADAPT | Establish a BDD-X reference run with paper-style caption/control metrics. | Usable baseline. Earlier reproduction reached much stronger text metrics than later ACPR V2 attempts. | Do not compare a new module to paper numbers before first comparing it against this local reproduction checkpoint/history. |
| FlowTrace PMT V1 |
ate_x_flowtrace_pmt_v1_worktree | Add traffic-flow/PMT mechanism while preserving ADAPT training semantics. | Required hard smoke limit fixes and strict doc package. | --max_steps alone was not a true training-loop cap; hard-stop plumbing must be verified from log lines. |
| ACPR FlowCalPP V1 | FATE-X ACPR worktree family | Add traffic-flow-aware control/text outputs and ADAPT-aligned evaluation. | Added epoch-end eval/resume/best checkpoint logic. | Any ADAPT-aligned trainer must evaluate test split every epoch and update checkpoint from real metrics, not a placeholder. |
| ACPR FlowCal V2 | cpr_flowcal_v2 path | Continue from a strong previous checkpoint and prove new modules improve it. | Failed to beat the previous ADAPT/ACPR baseline. Text stayed around CIDEr_des+exp=2.05-2.08; local ADAPT reproduction had stronger values. | New module schedules are not useful if resume/eval bridge cannot reproduce the starting checkpoint before training. |
| DynFlow V1 package | E:/sbw/FATE_Drive/fate_x_acpr_dynflow_v1_worktree, branch cpr_dynflow_v1 | Implement a stricter direct-image ACPR-DynFlow path with Video Swin, BERT, OIA predicates, gradient-chain audit, and preflight gates. | Code implemented and preflight passed once at commit 8d4b7ca, but a later train/eval honesty patch is currently uncommitted and invalidates the previous review pass until re-run. | Review pass is tied to a Git SHA; any code or doc commit after a pass requires another final audit/pass. |

### Current DynFlow V1 implementation contract
- Formal input must remain direct BDD-X frames [B,32,3,224,224]; no cached ADAPT features/logits as formal input.
- Allowed initializers are generic BERT-base, Kinetics Video Swin, and the BDD-OIA ACPR-CalAlign predicate-query checkpoint.
-
ate_x.engine.train_acpr_dynflow and
ate_x.engine.eval_acpr_dynflow are the official trainer/evaluator.
- Preflight and audit reports must pass before formal training; review pass must be regenerated after the latest commit.
- Training must not silently use fake text scores or save checkpoint_best_text.pth from a placeholder.

### Current action checklist
- [x] Preserve one unified set of three records instead of creating new scattered MD files.
- [x] Record V2 failure against local ADAPT reproduction, not only against paper numbers.
- [x] Record OIA checkpoint resolution and gradient-chain root causes.
- [x] Record that eval_acpr_dynflow.py / 	rain_acpr_dynflow.py had a fake/absent text-eval gap after the earlier review pass.
- [ ] Commit the latest train/eval honesty patch plus this record expansion.
- [ ] Re-run DynFlow preflight and audit on the new HEAD before claiming formal review-pass validity.
- [ ] Only start formal training after the new review pass confirms current HEAD and text/control eval paths are valid.

### Hard stop conditions for future runs
- If a run cannot reproduce the selected starting checkpoint metrics through the new eval bridge, do not continue staged training.
- If control metrics are scale-broken, e.g. course RMSE jumps from single digits to around 89, fix the eval/data bridge before training.
- If generated-text metrics are unavailable, do not save a best-text checkpoint.
- If a new review pass was generated before a later commit, treat it as stale.
### Post-commit DynFlow final preflight evidence - 2026-06-23 05:17:35 +08:00
- Checked HEAD before recording: $head.
- GitHub branch state before recording: $remote.
- Final preflight/audit directory inspected: .background_runs/acpr_dynflow_v1_final_preflight_20260623_0515.
-
eview_report.json: passed=true, lockers=[], missing_reports=[],
ailed_reports=[].
- git_provenance.json: branch cpr_dynflow_v1, local HEAD equals GitHub HEAD, clean worktree.
- oia_predicate_transfer_audit.json: oia_loaded=true, source model, source dim 384, checkpoint SHA 84d3744a7505cca19b33ac2b517b58d71c98fd580f162dec4a6eee2aee1f64b2.
- gate_gradient_chain.json: passed=true, missing_components=[], missing_trainable_grad_params=0,
rozen_params_with_grad=[].
- Important: this documentation append changes Git HEAD after the 0515 pass. Therefore a new final preflight/audit must be run again after committing this section; the latest run directory, not this paragraph alone, is the binding proof.
## 2026-06-23 ACPR-DynFlow V1 训练停止与下一步计划修正

本轮目标原本是启动 ACPR-DynFlow V1 full training 并观察能否在 ADAPT/FlowCalPP 基础上取得稳定提升。但实际训练过程中出现两个必须记录的结论：第一，旧 formal run 的 explanation supervision 没有真正生效；第二，修复后 full run 单 epoch 时间远超可接受范围，当前配置不适合继续完整长训。

本轮已经完成的计划修正如下：

1. 停止无效旧 run。旧 commit `50505c1` 的训练日志中 `explanation_text=0.0`，说明解释文本损失没有进入有效监督。该 run 不能作为有效实验结果继续承接。
2. 修复文本监督落地问题。`fate_x/acpr_dynflow/text_decoder.py` 已修正 `masked_pos` 解析逻辑：ADAPT/BDDX dataloader 提供的是二值 mask `[B, 30]`，不是显式 token 位置列表；现在会先转换为真实 token positions，再用 packed `masked_ids` 计算 action/explanation 两段 loss。
3. 新增回归测试。`tests/acpr_dynflow/test_text_decoder_masked_positions.py` 覆盖二值 `masked_pos` 情况，防止 explanation half 再次被静默跳过。
4. 修复后 smoke 与 formal run 都确认 `explanation_text` 非零，说明文本监督已经恢复。
5. 正式 full run 已停止。停止点为 epoch 0 batch `254/1639`，尚未完成第一轮，因此没有新的 checkpoint 或 eval 结果。

训练时长问题的计划判断：

- 当前配置的 `optimizer_steps_per_epoch=235` 是优化器步数，不是 dataloader micro-batches。
- 实际每个 epoch 是 `1639` 个 micro-batches；batch size 为 `10`，gradient accumulation 为 `7`，有效 batch 约 `70`。
- 该 run 在单卡 48G 上显存占用约 `45.6G/49.1G`，但吞吐仍然较低；按 batch 推进速度估计，单 epoch 约 `18-21` 小时。
- 因此直接按当前 full Video Swin + 32f + online decode/eval 跑 20 epoch 不现实，继续训练会消耗大量时间但不能快速给出有效结论。

后续计划必须调整为：

1. 以后任何 DynFlow full training 必须以 `e66bce98285883315b949250dd73ec17df3f3214` 之后的代码为最低起点，不能回到 `50505c1` 或更早的文本监督实现。
2. 不再直接启动 20 epoch 原配置长训；先做更短的 bounded run 或缓存特征路线。
3. 优先评估加速方案：冻结/缓存 Video Swin 特征、减少在线重复视频编码、使用短 epoch smoke + eval、或调整数据读取与预取，而不是盲目加 batch。
4. 如果目标是验证 DynFlow 模块是否有效，应先在可控计算预算内跑完至少一个完整 epoch 并完成 ADAPT-aligned eval，再谈是否长训。

## 2026-06-24 ACPR-DynFlow-Swin V1 contract installation start

- Installed formal runbook/config/manifest/audit skill from user package.
- Verified current worktree and branch before code changes.
- Formal implementation is not yet complete; training remains blocked until preflight review pass.

## 2026-06-24 20:24:20 - ACPR DynFlow Swin V1 remaining gates before formal training
- Connect and verify the real BDD-X 32f image dataloader for python -m fate_x.engine.train_acpr_dynflow_swin; smoke batches are not acceptable for formal results.
- Run the throughput probe on the real dataloader and enforce the plan gate: projected epoch time must not exceed 4h.
- Generate the formal review pass file only after import graph, config binding, tensor contract, real-data smoke, checkpoint save/load, and throughput gates all pass.
- Keep formal launch foreground-supervised only; no Start-Process, schtasks,
ohup, or detached runners.



## 2026-06-24 20:36:59 - Updated formal readiness status
- Completed: real BDD-X train loader reaches the formal Swin model for at least one batch.
- Remaining: run GPU throughput probe on real dataloader; enforce projected epoch <= 4h before formal training.
- Remaining: expand evaluator from scaffold metrics to full ADAPT legal text/control metrics on test split.
- Remaining: create review pass file only after real-data preflight, throughput, evaluation, checkpoint reload, and visual/faithfulness smoke all pass.


## 2026-06-24 20:40:00 - Remaining formal training gate
- Run `python -m fate_x.engine.probe_acpr_dynflow_swin_throughput` on the intended GPU/WSL environment with real BDD-X data.
- Do not create `REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt` until the real GPU probe passes the 4h projected epoch gate and evaluator smoke is upgraded beyond scaffold selection logic.

## 2026-06-24 22:33:56 ACPR-DynFlow-Swin V1 strict gate plan update
- 当前要求：不新建新 md；所有记录继续写入 ACPR_FlowCalPP 三份过程文档。
- 训练启动条件：必须先通过 WSL/Linux CUDA direct-image smoke、ADAPT metric parity、OIA/nnPU/CalAlign/mass/intervention/Canvas/Atlas 动态 gate、100-step throughput；且 projected train epoch <= 2 hours。
- 当前禁止：未过 gate 前不杀旧进程、不启动当前训练、不生成 REVIEW_PASS。


## 2026-06-24 23:06:02 ACPR-DynFlow-Swin V1 updated task gate

Current status: **blocked for formal training**.

Completed in this session:

- Fixed WSL git preflight provenance fallback for Windows gitdir paths.
- Fixed blocking audit so missing output_dir cannot accidentally pass dynamic gate review.
- Fixed formal model native final-stage reshape bug.
- Fixed formal model Swin-final-to-motion dimension bug.
- Verified real WSL/CUDA direct-image smoke passes one real BDD-X batch with nonzero gradients.
- Ran 100-step real CUDA throughput probe; current estimate is `<2h/epoch` under the existing probe script.

Must still be completed before formal training may start:

- ADAPT metric parity using real ADAPT reference predictions/evaluator.
- OIA predicate transfer dynamic evidence with checkpoint/key/SHA/order/gate report.
- nnPU/CalAlign with real positive/reliable-negative/unlabeled counts and nonzero loss.
- Mass conservation and exact ledger identity within required tolerance in dynamic gate reports.
- Real intervention recompute from earliest affected layer, not display tensor masking.
- Real Canvas/Atlas outputs bound to tensor evidence.
- Full preflight reports all marked pass on a clean pushed SHA.
- Review pass file bound to the exact clean local/GitHub SHA.

Training rule preserved: do not launch training until all required gates pass. If gates pass, only launch when projected epoch time remains below the user limit of two hours.


## 2026-06-24 23:12:07 ACPR-DynFlow-Swin V1 launch decision

Decision: **do not launch training**.

Reason:

- User gate `epoch time <= 2h` is currently plausible under the existing batch=1 100-step probe (`1.44h/epoch`), but formal review pass is absent.
- The formal plan says training may start only after all blocking gates and review pass bind to the clean pushed SHA.
- Current clean SHA `835065fce8b302c47976a0cef614a85e10c0248a` still has dynamic mechanism blockers, so launching would violate the plan.

Next required engineering work before any launch:

1. Replace placeholder preflight reports with executable dynamic gates.
2. Implement nonzero nnPU/CalAlign evidence and loss.
3. Prove ADAPT metric parity and autoregressive evaluation.
4. Implement real intervention recompute and Canvas/Atlas evidence.
5. Re-run preflight/audit on a clean pushed SHA and only then launch training.
