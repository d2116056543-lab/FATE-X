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
