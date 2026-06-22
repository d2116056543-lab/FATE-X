<!--
Canonical ACPR FlowCalPP / FlowCal V2 Progress ledger.
This file intentionally contains both the earlier V1/FlowCalPP record and the later V2 record.
Restored and merged from git commit ccb0370 on 2026-06-23 00:32:59 Asia/Shanghai after the user requested one continuous three-file history.
Do not split V2 into separate task/findings/progress files again.
-->

# Unified ACPR FlowCalPP / FlowCal V2 Progress Ledger

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

# ACPR FlowCalPP Progress

更新时间：2026-06-21 14:25 Asia/Shanghai

## 1. 时间线总览

### 2026-05：ADAPT 复现与 BDD-X 数据确认

- 远端 ADAPT repo：`E:\sbw\ADAPT_repro\ADAPT`
- processed 数据：`E:\sbw\ADAPT_repro\ADAPT\ADAPT_PREPROCESSED_DATASET`
- 用户下载了作者提供的 processed 数据。
- 确认 `datasets` 和 `datasets_part` 都属于 BDD-X processed 数据结构。
- `datasets/BDDX/*_32frames.yaml` 对应图像与 caption 样本。
- `datasets_part/BDDX/*_32frames.yaml` 对应连续车辆控制信号。
- 单 GPU 可训练，但速度显著慢于论文多卡设置。
- 后续补齐 latest/best ckpt 与 scheduler-correct resume 逻辑。

经验：

- BDD-X explanation 文本是自然语言，不是固定类别标签，ADAPT 用 COCO caption 指标评估。
- BDD-X control 是连续信号，不能简单按 PSI 那类 maintain/slow/stop 评价。

### 2026-06-18：FlowTrace PMT V1

- 根据 FlowTrace PMT V1 计划实现视频流/路径机制相关代码。
- 重点解决“能接入训练”的问题。
- 后续发现 smoke 不够，必须严格审查：
  - 是否真的 hard-stop。
  - 是否有 per-epoch eval。
  - 是否有 best checkpoint。
  - 是否与 ADAPT metric contract 对齐。

经验：

- `--max_steps` 曾经只限制样本抽取，不限制真实训练 loop。这类 smoke 必须看真实 `global_step` stop line。
- 能训练一个 batch 不等于功能落地完整。

### 2026-06-19：ACPR FlowCalPP V1 初版实现

涉及代码范围：

- `configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml`
- `fate_x/acpr_flow/model.py`
- `fate_x/acpr_flow/reason_control_adapter.py`
- `fate_x/acpr_flow/temporal_seca.py`
- `fate_x/engine/acpr_bddx_data.py`
- `fate_x/engine/train_acpr_flowcal_pp.py`
- `fate_x/engine/acpr_action_text_eval.py`
- `fate_x/engine/acpr_control_eval.py`
- `fate_x/losses/acpr_flowcal_losses.py`
- `src/tasks/run_adapt.py`

测试覆盖范围：

- config contract
- formal path integration
- hardpair
- control eval
- traffic audit
- epoch eval/resume
- dataloader workers
- eval safety
- ADAPT CIDEr eval

经验：

- 严格计划不能只看文件名存在，要验证每个计划功能是否真实进入 forward/loss/eval/checkpoint。
- 复杂机制必须用测试证明梯度路径，不然很容易“有 loss 字段但没训练到对应模块”。

### 2026-06-20：按 ADAPT 对齐 eval 和 checkpoint

用户明确要求：

- 训练必须和 ADAPT 的数据/指标对齐。
- 每轮都要 test eval。
- 保存 best ckpt。
- 后台跑但要前台监督。
- 不要因为 eval 问题浪费一轮训练。

已修改：

- 每轮 eval 前先保存 `checkpoint_latest.pth`。
- 添加 ADAPT/COCO caption eval。
- 添加 continuous speed/course control eval。
- 添加 `checkpoint_best_text.pth`、`checkpoint_best_control.pth`、`checkpoint_best_adapt_joint.pth`、`checkpoint_best_test.pth`。
- 原离散 decision proxy 被降级，不再作为主评价或 best 选择依据。

经验：

- test metric 更新 best checkpoint 必须用真实 test output，不应只用 train/val loss。
- ADAPT 文本指标和 control 指标是两套评价，不应互相替代。

### 2026-06-20：historical text best

Run：

`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_resume_epoch5_foreground_20260620_185230\train`

Best file：

`eval_epoch_005.json`

结果：

- `CIDEr_des = 1.4878`
- `CIDEr_exp = 0.7872`
- `CIDEr_des+exp = 2.2750`
- `speed RMSE = 2.6372`
- `course RMSE = 6.1163`

问题：

- 该时期还有旧离散 proxy 字段。
- 不能作为 current strict traffic/circular-control 机制最终结论。

经验：

- 后续比较必须同时写 run directory 和 eval file，不允许只说“最好的结果”。

### 2026-06-21：strict epoch14 完整评价

Run：

`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_resume_envfix_20260621_065540\train`

Eval：

`eval_epoch_014.json`

结果：

- `CIDEr_des = 0.6488`
- `CIDEr_exp = 0.5441`
- `CIDEr_des+exp = 1.1930`
- `speed RMSE = 2.5398`
- `course RMSE = 6.1176`
- `speed Acc@0.5 = 0.2894`
- `course Acc@0.5 = 0.8676`

新增输出：

- `pred_speed_delta_corr`
- `pred_course_delta_corr`

意义：

- 修复了“只能证明交通流因子和标签相关，不能证明和模型预测相关”的不足。
- 但这仍是相关性，不是 zero-out 因果证据。

### 2026-06-21：course circular fix run

Run：

`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030\train`

状态：

- 按用户要求停止。
- 停止在 epoch 15 中途。
- 没有新 eval。
- 有效 ckpt：`checkpoint_latest.pth`
- 无效 partial：`checkpoint_latest.pth.tmp`

最后 batch：

- loss：`0.7541`
- action_text：`0.1764`
- explanation_text：`0.4486`
- control：`1.1019`
- predicate_pu：`0.4291`
- flow_pu：`0.2880`
- reason_semantic：`0.5427`
- future_control：`0.7403`
- hardpair_active_pair_rate：`1.0`
- hardpair_candidate_count：`4096`

结论：

- 这个 stopped run 只能作为续训点，不能作为新结果。
- 必须跑完一个完整 epoch 并产出 eval 后才能判断 course circular fix 效果。

## 2. 代码修改与原因链

### 2.1 ADAPT-aligned caption eval

问题：

- 训练日志没有真正对齐 ADAPT 的文本评价，只能看到训练 loss 或不完整 CIDEr。

修改：

- 增加按 `description` / `explanation` 分离的 ADAPT/COCO caption eval。
- 生成 `des_metrics`、`exp_metrics`、`CIDEr_des_plus_exp`。

结果：

- `eval_epoch_014.json` 和 `eval_epoch_005.json` 均可直接对比 ADAPT 文本指标。

### 2.2 Continuous control eval

问题：

- 用户任务需要车辆控制；只看文本不够。
- 早期尝试过离散 decision proxy，但不符合 BDD-X 原始连续控制标注。

修改：

- 使用 `datasets_part/BDDX/testing_32frames.yaml` 读取 speed/course 连续信号。
- 输出 RMSE、MAE、Acc@threshold。

结果：

- control 指标接近 ADAPT。
- 离散 proxy 不再作为主指标。

### 2.3 Checkpoint 逻辑

问题：

- 如果 eval 报错，该轮训练成果可能没保存。
- 一个 best 文件无法表达 text/control/joint 多目标。

修改：

- eval 前保存 latest。
- 分离 best_text / best_control / best_adapt_joint / best_test。

结果：

- strict run 中可见多个 best ckpt。
- stopped run 有完整 latest 可续训。

### 2.4 Traffic-flow audit

问题：

- 初始 audit 只能看 traffic factor 和真实 control delta 的相关性。
- 用户指出这不能证明模型真的用这些因子驱动预测。

修改：

- 增加 predicted control delta 统计。
- 输出 `pred_speed_delta_corr` 和 `pred_course_delta_corr`。

结果：

- strict epoch14 中多个 factor 的 pred corr 非 null。
- 仍需 zero-out/counterfactual 验证因果影响。

### 2.5 Course circular fix

问题：

- course 是角度类信号，线性差分会受 wrap-around 影响。

修改：

- 引入 circular delta 处理。

状态：

- 代码已改。
- 当前 run 被中途停止，尚无新 eval。

### 2.6 Hardpair formal path

问题：

- 测试发现 `model.hardpair.proj.weight.grad is None`。
- 表面上日志有 hardpair loss，但 formal-path 测试证明投影层未必有梯度。

根因：

- `RecordingCaptioningModel` 没有 BERT word embedding。
- `_caption_action_target_embedding()` fallback 使用 predicted state。
- predicted action embedding 与 queued action target 不够相似，导致无 eligible hard pair。
- raw loss 退化成不经过 projection 的 zero tensor。

修复：

- fallback 改用 `reason_for_pair` target。
- 真实 ADAPT captioning 有 BERT embedding 时仍走 action-text embedding。

验证：

- 单项 hardpair formal-path 测试通过。
- 全套 ACPR 测试通过。

## 3. 环境与运行问题记录

### 3.1 Windows vs Linux/WSL

问题：

- Windows 远端可以跑，但训练/评估速度慢。
- 原文/ADAPT 更接近 Linux 环境。

处理：

- 使用 WSL `ADAPT-Ubuntu` 和 `/opt/conda/envs/adapt/bin/python`。
- 训练 batch/workers 调整为更实用的单卡配置：batch 4、workers 4。

经验：

- Linux 命令不能直接写在 Windows PowerShell 里，要通过 `wsl -- bash -lc`。

### 3.2 SSH/PowerShell quoting

问题：

- 多次复杂 one-liner 造成远端 `Set-Location` 不生效或命令被本地解释。
- 典型现象：`git status` 报 `not a git repository`，实际是在本地 cwd 或错误目录。

处理：

- 后续关键验证改用 EncodedCommand、临时 `.ps1` 或明确的 remote script。

经验：

- 远端验证不能依赖超长嵌套引号。

### 3.3 Git hygiene

问题：

- Windows/patch 混写导致 CRLF 和 trailing whitespace。
- `git diff --check` 初次失败。

处理：

- 对本次 staging 范围统一 LF。
- 清理尾随空白。
- 重新运行 `git diff --check`。

经验：

- 每次 commit 前必须运行 `git diff --check`。

## 4. 验证记录

### 4.1 Hardpair 单项验证

命令：

`pytest tests/test_acpr_flow_formal_path_integrations.py::test_hardpair_loss_is_integrated_into_model_and_optimizer_group -q`

结果：

- 通过。

### 4.2 ACPR 全套相关测试

命令：

`CUDA_VISIBLE_DEVICES='' /opt/conda/envs/adapt/bin/python -m pytest tests/test_acpr_action_text_eval.py tests/test_acpr_flow_adapt_cider_eval.py tests/test_acpr_flow_control_eval.py tests/test_acpr_flow_control_temporal_path.py tests/test_acpr_flow_dataloader_workers.py tests/test_acpr_flow_epoch_eval_resume.py tests/test_acpr_flow_eval_safety.py tests/test_acpr_flow_traffic_audit.py tests/test_acpr_flow_config_contract.py tests/test_acpr_flow_formal_path_integrations.py tests/test_acpr_flow_hardpair.py -q`

结果：

- `43 passed, 1 skipped, 6 warnings in 36.94s`

### 4.3 Git whitespace

命令：

`git diff --check`

结果：

- 通过。

### 4.4 GitHub push

已推送：

- `7f9f8af Add ACPR FlowCalPP eval records and control audit fixes`
- `f7d7815 Document ACPR FlowCalPP run stop and validation findings`

当前还需要把本次“更详细记录”追加 commit 并推送。

## 5. 当前有效文件

远端：

- `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree\docs\acpr_flowcalpp\ACPR_FlowCalPP_task_plan.md`
- `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree\docs\acpr_flowcalpp\ACPR_FlowCalPP_findings.md`
- `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree\docs\acpr_flowcalpp\ACPR_FlowCalPP_progress.md`

本地：

- `E:\FATE_X_ACPR_FlowCalPP_Records\ACPR_FlowCalPP_task_plan.md`
- `E:\FATE_X_ACPR_FlowCalPP_Records\ACPR_FlowCalPP_findings.md`
- `E:\FATE_X_ACPR_FlowCalPP_Records\ACPR_FlowCalPP_progress.md`

有效续训 ckpt：

- `E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030\train\checkpoint_latest.pth`

无效文件：

- `E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030\train\checkpoint_latest.pth.tmp`

## 6. 下一步如果继续训练

1. 删除或忽略 `.tmp`。
2. 从有效 latest resume。
3. 跑完至少一个完整 epoch。
4. 确认产出新的 `eval_epoch_XXX.json`。
5. 对比 course circular fix 后的 control metric。
6. 检查 text CIDEr 是否继续低于 historical best。
7. 如果低于 historical best，先查文本分支和 loss 权重，不要继续只增强交通流分支。
8. 如果 traffic audit 要证明机制有效，下一步必须做 zero-out 或 counterfactual audit。


---

# Part B: Detailed V2 Record Appended To The Same Ledger

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

## ACPR-DynFlow V1 chronological progress (2026-06-23 03:45 China time)

1. Read and installed the package files from `FATE_X_ACPR_DynFlow_V1_Codex_Package_20260622`.
2. Audited the source `flowtrace_pmt_v1` worktree and moved untracked scratch files into a timestamped ignored snapshot instead of deleting them.
3. Created target worktree `E:/sbw/FATE_Drive/fate_x_acpr_dynflow_v1_worktree` and branch `acpr_dynflow_v1`.
4. Pushed branch `acpr_dynflow_v1` to `github=https://github.com/d2116056543-lab/FATE-X.git` at base commit `8a52f4e99e81406cd949afaadb16c7483cf5025d`.
5. Generated the DynFlow namespace, config files, engine entrypoints, explain modules, scripts, and tests.
6. Initial test failure: `MesoscopicLaneFlow` used 31 relative-motion frames against 32 occupancy frames.
7. Fix: prepended a zero relative-motion row so `rel_motion` aligns with full 32-frame occupancy.
8. Re-ran tests after the fix: `48 passed in 53.18s`.
9. Ran preflight. First run had a self-audit bug where `review_report.json` was counted missing before it was written.
10. Fixed `audit_acpr_dynflow.py` so `review_report.json` is not included in its own pre-write missing list.
11. Re-ran preflight: `missing_reports=[]`; remaining blockers: `dirty_worktree`, `oia_checkpoint_unresolved`.
12. Ran synthetic training smoke. It emitted `ACPR_DYNFLOW_BATCH` and verified direct image shape `[1, 32, 3, 224, 224]`.
13. Fixed `eval_acpr_dynflow.py` to use `torch.load(..., weights_only=False)` explicitly and eliminate the warning observed during smoke evaluation.
14. Re-ran compile/smoke after warning fix: batch output remained normal and no `torch.load` FutureWarning was observed.

### Next exact steps
1. Commit and push the current implementation branch after final `git diff --check`.
2. Re-run preflight after commit; expected hard blocker should be only `oia_checkpoint_unresolved`.
3. Obtain or build the required BDD-OIA ACPR-CalAlign predicate-query checkpoint.
4. Replace `oia_acpr_checkpoint: UNRESOLVED_REQUIRED_FATE_OIA_ACPR_CALALIGN_V1_2_QUERY_CHECKPOINT` in `configs/acpr_dynflow_v1_bddx_32f_224.yaml`.
5. Run formal preflight again and require `passed=true` plus `REVIEW_PASS_ACPR_DYNFLOW_V1.txt`.
6. Only then start formal BDD-X training under the foreground supervisor.

## ACPR-DynFlow V1 real-loader progress (2026-06-23 04:16 China time)

1. Confirmed `torchvision` and `transformers` are installed in `sbw39`.
2. Downloaded official Kinetics-600 Video Swin checkpoint to the configured plan path.
3. Downloaded Hugging Face `bert-base-uncased` to the configured plan path.
4. Inspected official Video Swin checkpoint structure and confirmed it contains `353` state-dict keys under `backbone.*`.
5. Compared against torchvision `swin3d_b` and confirmed tensor count/shape compatibility despite key naming mismatch.
6. Replaced lightweight `ACPRDynFlowVideoBackbone` with actual `swin3d_b` feature extraction and deterministic official-to-torchvision key conversion.
7. Replaced lightweight text decoder path with BERT-base top-4 trainable decoder path.
8. Patched `model.py` to pass `paths.bert_dir` into `DynFlowTextDecoder`.
9. Patched preflight/audit so formal review fails if Swin/BERT are missing or not actually loaded.
10. Fixed BERT protobuf import issue with local environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
11. Ran CUDA synthetic smoke and real-loader preflight; `bert_not_loaded` disappeared.
12. Ran full tests after the real-loader changes: `48 passed, 102 warnings in 295.85s`.
13. Next step is commit/push this hardening, then rerun preflight from a clean worktree. Expected blocker after commit: only `oia_checkpoint_unresolved`.

## 2026-06-23 04:45 进度记录：ACPR-DynFlow V1 strict gate 修复

1. 定位 OIA blocker：`ckp/classifier.pth.tar` 只有 linear classifier，不含 predicate queries；转向 `fate_oia_acpr_calalign_v1_2_worktree` 搜索真实 CalAlign checkpoint。
2. 找到并采用：`E:/sbw/FATE_Drive/fate_oia_acpr_calalign_v1_2_worktree/.background_runs/acpr_calalign_v1_2_resume_e15_17_sched28_20260616_125105/checkpoint_best_test_final_calibrated.pth`，其中 top key 有 `model/optimizer/epoch/metrics/base_lrs`，`model` 内含 `predicate_head.predicate_queries`。
3. 改写 `predicate_transfer.py`：加载 checkpoint、抽取 32 条 predicate query、支持 384->256 mapper、保留 name prior 和 trainable residual，并在 audit 中写 source/shape/SHA。
4. 改写 preflight/audit：动态 forward 后写 `oia_loaded`；formal audit 对 `oia_loaded=false` 阻断；required report 中任何 `passed:false` 也阻断。
5. 首次动态 preflight 结果：OIA 已通过，但 `gate_gradient_chain.passed=false`，且旧 review 没阻断 failed report，暴露 gate wiring 问题。
6. 排查 gradient missing params：`lane_flow.encoder`、`reasoner.lateral`、`backbone.text_proj`、`swin.head`、`bert.pooler`。
7. 修复真实 dead path：lane-flow token 接入 traffic-state factor，lateral bias 接入 course decision ledger，visual text token 接入 text decoder；未用 head/pooler 冻结。
8. 二次动态 preflight：`failed_reports=[]`，`gate_gradient_chain.passed=true`，唯一 blocker 为 `dirty_worktree`。
9. 验证命令：`python -m compileall -q fate_x tests/acpr_dynflow` exit 0；`python -m pytest tests/acpr_dynflow -q` 48 passed；`git diff --check` exit 0。
10. 当前状态：等待 commit/push，然后 clean HEAD 重新跑 full preflight + review pass。未写 `REVIEW_PASS` 前，formal training 仍不得启动。

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

## Unified Progress Expansion - 2026-06-23 05:09:06 +08:00

### 1. ADAPT and BDD-X foundation
- Verified that the BDD-X preprocessed package is the practical data base for ADAPT-style reproduction.
- Established that datasets and datasets_part have different roles in the local ADAPT setup; future comparisons must preserve the actual reference data path rather than substituting a different split silently.
- Recorded that local ADAPT reproduction is the immediate benchmark for ACPR modules; paper numbers are secondary unless exact protocol parity is proven.

### 2. FlowTrace / FlowCalPP / V1 engineering lessons
- Added real epoch-end evaluation and best-checkpoint logic after discovering that naming a checkpoint est_test is not sufficient unless driven by actual test metrics.
- Added resume state handling and monitoring artifacts so SSH disconnects do not destroy the run state.
- Fixed smoke/step-limit behavior in earlier FlowTrace work after discovering that a nominal max-step argument did not cap the real ADAPT loop.
- Lesson retained: for any future package, a smoke must prove the actual train loop stops/evaluates where intended, not merely that arguments are parsed.

### 3. ACPR FlowCal V2 run outcome
- V2 was intended to continue from a stronger prior checkpoint and verify that added flow/control modules improve the previous result.
- Observed result did not improve over the local ADAPT reproduction baseline. Recorded text metrics stayed around CIDEr_des+exp=2.05-2.08, below the earlier reproduction history.
- Control/course metrics were not normal; the course RMSE scale mismatch required stopping rather than continuing.
- Decision made: do not keep training V2 until the resume checkpoint can be evaluated through the V2 bridge and reproduce the pre-V2 baseline.

### 4. DynFlow V1 package implementation chronology
- Created branch/worktree cpr_dynflow_v1 under E:/sbw/FATE_Drive/fate_x_acpr_dynflow_v1_worktree.
- Implemented package namespace ate_x/acpr_dynflow and engine entrypoints for training, evaluation, preflight, audit, visualization, and supervision.
- Added direct-image data path with formal frame shape [B,32,3,224,224].
- Replaced placeholder visual/text paths with real Video Swin Kinetics and local BERT-base loaders.
- Located real OIA/CalAlign predicate checkpoint and wired predicate_head.predicate_queries into the initializer.
- Fixed gradient-chain implementation so intended modules receive gradients and frozen components remain frozen.
- Ran verification before the earlier commit chain: compileall, pytest tests/acpr_dynflow -q, git diff --check, synthetic preflight, formal audit, and review-pass generation.

### 5. Commits recorded before latest patch
- 9f8c122 Resolve ACPR DynFlow OIA and gradient gates.
- 8d4b7ca Record ACPR DynFlow final review pass.
- These commits were pushed to GitHub branch cpr_dynflow_v1 before the later train/eval honesty patch.

### 6. Latest discovered gap after the earlier review pass
- Inspection found that 	rain_acpr_dynflow.py still selected best-text checkpoints from a fake placeholder score.
- Inspection found that eval_acpr_dynflow.py did not yet emit real generated-caption ADAPT-style metrics.
- Patched files currently modified in the worktree:
  - ate_x/engine/eval_acpr_dynflow.py
  - ate_x/engine/train_acpr_dynflow.py
- Patch intent:
  - generate model caption rows during evaluation;
  - call the ADAPT-style caption evaluation bridge when real data is available;
  - write blockers when text metrics are unavailable;
  - prevent checkpoint_best_text.pth and joint checkpoint updates from fake text scores;
  - require a valid review pass tied to current Git HEAD before formal training.

### 7. Current not-yet-finished state
- The previous review pass is stale because code changed after it.
- The two train/eval files and this documentation expansion must be committed and pushed.
- After that, final dynamic preflight and formal audit must be rerun on the new clean HEAD.
- If the new preflight/audit fails, do not start formal training; record the blocker and fix it.

### 8. Next exact verification sequence
1. Commit code and documentation changes.
2. Push cpr_dynflow_v1 to d2116056543-lab/FATE-X.
3. Run python -m compileall -q fate_x tests/acpr_dynflow.
4. Run python -m pytest tests/acpr_dynflow -q.
5. Run git diff --check.
6. Run python -m fate_x.engine.run_acpr_dynflow_preflight --config configs/acpr_dynflow_v1_bddx_32f_224.yaml --output_dir <new_final_preflight_dir> --device cuda --synthetic.
7. Run python -m fate_x.engine.audit_acpr_dynflow --repo_root . --config configs/acpr_dynflow_v1_bddx_32f_224.yaml --output_dir <new_final_preflight_dir> --write_review_pass.
8. Confirm local HEAD, GitHub HEAD, and review-pass provenance all match.
9. Only after these checks, run formal training or a bounded real-data train/eval smoke.