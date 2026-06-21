# ACPR FlowCalPP Task Plan

更新时间：2026-06-21 13:20 Asia/Shanghai

## 当前状态

- 远端仓库：`E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree`
- GitHub branch：`flowtrace_pmt_v1` -> `https://github.com/d2116056543-lab/FATE-X/tree/flowtrace_pmt_v1`
- 当前训练已按要求停止：计划任务 `acpr_coursefix3_121030` 为 `Ready`，远端进程表没有 `coursecircularfix3/train_acpr_flowcal` 训练进程。
- 当前停止 run：`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030`
- 有效续训 ckpt：`...\train\checkpoint_latest.pth`，写入时间 `2026-06-21 12:46:50`。
- 无效临时文件：`...\train\checkpoint_latest.pth.tmp`，这是 epoch 15 中途停止时残留的 partial tmp，不能用于续训。
- 当前停止点：epoch `15`，global_step `64500`，optimizer_step `3717`，最后 batch loss `0.7541`。

## 固定记录文件

后续只维护这三份实验记录：

- `ACPR_FlowCalPP_task_plan.md`：任务范围、协议、路径、下一步计划。
- `ACPR_FlowCalPP_findings.md`：结果、组件作用、ADAPT 对比、问题根因。
- `ACPR_FlowCalPP_progress.md`：时间顺序日志、修改记录、运行状态。

本地同步目录：`E:\FATE_X_ACPR_FlowCalPP_Records\`。只同步这三份 MD。

## 实验协议

- 数据：BDDX 32-frame processed 数据，训练和测试沿用 ADAPT 结构。
- 文本评估：ADAPT/COCO caption 口径，分别评估 description 与 explanation，主文本指标为 `CIDEr_des + CIDEr_exp`。
- Control 评估：连续 control 口径，使用 speed/course 的 RMSE、MAE、Acc@0.1/0.5/1/5/10。
- 不再把 speed/course 强行离散化成 maintain/stop/straight/turn 作为主指标；旧离散 proxy 只保留为历史诊断说明。
- Checkpoint 逻辑：
  - `checkpoint_latest.pth`：每轮/保存点续训用。
  - `checkpoint_best_text.pth`：按 `CIDEr_des + CIDEr_exp`。
  - `checkpoint_best_control.pth`：按合法 continuous control 口径。
  - `checkpoint_best_adapt_joint.pth`：按 ADAPT 合法文本 + continuous control 综合口径。
  - `checkpoint_best_test.pth`：test metric 驱动的 best。

## 当前主线

1. 已完成：ADAPT 对齐评估、per-epoch test eval、best checkpoint 分类保存、traffic-flow audit、predicted control delta correlation 输出。
2. 已完成：删除/降级离散 decision proxy，不作为主评估。
3. 已完成：加入交通流语义因子审计，记录 target 与 predicted control delta 的相关性。
4. 已完成：针对 course wrap-around 问题做 circular delta 修复，当前 `coursecircularfix3` run 已启动后中途停止，尚未完成新一轮 eval。
5. 下一步：从 `checkpoint_latest.pth` 续训，至少跑完一个完整 epoch 以产生 course circular fix 后的新 eval，再比较 traffic audit 与 text/control 指标。
6. 下一步：若文本 CIDEr 长期不更新 best，优先检查 caption loss 权重、beam/max_len、ADAPT tokenizer/label 对齐，而不是继续只拉 control/traffic 分支。

## 续训注意事项

- 不要从 `.tmp` 文件恢复。
- 续训前先删掉或忽略 `checkpoint_latest.pth.tmp`，避免自动扫描脚本误用。
- 当前可恢复点是 `checkpoint_latest.pth`，不是 epoch 15 完整 eval。
- 任何新训练必须每轮评估前先保存 latest，以免 eval 报错导致该轮训练白跑。
