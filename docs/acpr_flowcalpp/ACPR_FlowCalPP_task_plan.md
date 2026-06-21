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
