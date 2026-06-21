# ACPR FlowCalPP Progress

更新时间：2026-06-21 13:20 Asia/Shanghai

## 时间线

### 2026-05：ADAPT 复现与数据确认

- 数据路径：`E:\sbw\ADAPT_repro\ADAPT\ADAPT_PREPROCESSED_DATASET`。
- 确认 `datasets` 与 `datasets_part` 均来自 BDD-X processed 数据；训练/评估沿用 32-frame 结构。
- 单卡训练已验证可以运行；保存 latest/best 和续训逻辑后续被补齐。

### 2026-06-18：FlowTrace PMT V1

- 按 FATE-X FlowTrace PMT V1 计划实现视频流/路径相关模块。
- 该阶段主要解决“能接入训练”的问题，后续发现不能只看 smoke，需要严格检查 eval、ckpt 和指标口径。

### 2026-06-19：ACPR FlowCalPP V1 实现

- 按 ACPR FlowCalPP V1 计划修改：`configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml`、`fate_x/acpr_flow/*`、`fate_x/engine/train_acpr_flowcal_pp.py`、loss 与测试。
- 增加/修改测试覆盖：config contract、formal path integration、hardpair、control eval、traffic audit、epoch eval/resume 等。
- 后续发现只说“能训练”不够，必须与 ADAPT 的 test eval 与 checkpoint 行为对齐。

### 2026-06-20：ADAPT-aligned eval 与 checkpoint 修复

- 加入每轮 test eval，输出 ADAPT/COCO caption 指标。
- 加入 continuous speed/course control eval，替换不合法的离散 action proxy 主评估。
- 增加 checkpoint 分类：latest、best_text、best_control、best_adapt_joint、best_test。
- 增加 eval 前保存 latest，防止 eval 出错导致训练成果丢失。

### 2026-06-20：历史文本最佳 run

- Run：`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_resume_epoch5_foreground_20260620_185230\train`。
- Best eval：`eval_epoch_005.json`，epoch `5`。
- CIDEr_des `1.4878`，CIDEr_exp `0.7872`，CIDEr_des+exp `2.2750`。
- Speed RMSE `2.6372`，Course RMSE `6.1163`。
- 注意：这个 run 仍包含旧离散 decision proxy 字段，不能作为当前 strict mechanism 的唯一结论。

### 2026-06-21：strict epoch14 run

- Run：`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_resume_envfix_20260621_065540\train`。
- Eval：`eval_epoch_014.json`，epoch `14`。
- CIDEr_des `0.6488`，CIDEr_exp `0.5441`，CIDEr_des+exp `1.1930`。
- Speed RMSE `2.5398`，Course RMSE `6.1176`。
- Traffic audit 已包含 `pred_speed_delta_corr` 和 `pred_course_delta_corr`，说明修复了“只能看标签相关、不能看模型预测相关”的不足。

### 2026-06-21：course circular fix run

- Run：`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030\train`。
- 停止点：epoch `15`，global_step `64500`，optimizer_step `3717`。
- 最后 batch loss `0.7541`。
- 最后 batch loss components：
  - action_text `0.1764`，explanation_text `0.4486`。
  - control `1.1019`，future_control `0.7403`。
  - flow_pu `0.2880`，predicate_pu `0.4291`。
  - reason_semantic `0.5427`，hardpair_active_pair_rate `1.0000`。
- 该 run 被手动停止在 epoch 中途，没有新 eval；有效 checkpoint 是 `checkpoint_latest.pth`，临时 `.tmp` 无效。

## 代码变更范围

生产代码/配置：

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

测试：

- `tests/test_acpr_action_text_eval.py`
- `tests/test_acpr_flow_adapt_cider_eval.py`
- `tests/test_acpr_flow_control_eval.py`
- `tests/test_acpr_flow_control_temporal_path.py`
- `tests/test_acpr_flow_dataloader_workers.py`
- `tests/test_acpr_flow_epoch_eval_resume.py`
- `tests/test_acpr_flow_eval_safety.py`
- `tests/test_acpr_flow_traffic_audit.py`
- `tests/test_acpr_flow_config_contract.py`
- `tests/test_acpr_flow_formal_path_integrations.py`
- `tests/test_acpr_flow_hardpair.py`

临时排查脚本未列为正式代码，除非后续明确整理进 `scripts/` 或 `tools/`。

## 提交前验证记录

- `git diff --check` 初次发现 CRLF/尾随空白污染；已对本次提交范围规范化为 LF 并清理尾随空白。
- ACPR 测试集合第一次结果：42 passed, 1 skipped, 1 failed。失败项是 hardpair projection 梯度为 `None`。
- hardpair 根因：测试降级 captioning model 没有 BERT word embedding，action embedding fallback 使用 predicted `bundle.global_reason_state`，导致与队列 action embedding 相似度不足，没有 eligible hard pair，raw loss 变成不经过 projection 的零张量。
- 修复：`action_for_pair = self._caption_action_target_embedding(batch, reason_for_pair)`。真实 ADAPT captioning 有 BERT embedding 时仍走 action-text embedding；无 BERT embedding 的测试/降级路径用 target fallback 保证 hardpair 梯度路径有效。
- 单项验证：`tests/test_acpr_flow_formal_path_integrations.py::test_hardpair_loss_is_integrated_into_model_and_optimizer_group` 通过。
- 完整验证：`pytest tests/test_acpr_action_text_eval.py tests/test_acpr_flow_adapt_cider_eval.py tests/test_acpr_flow_control_eval.py tests/test_acpr_flow_control_temporal_path.py tests/test_acpr_flow_dataloader_workers.py tests/test_acpr_flow_epoch_eval_resume.py tests/test_acpr_flow_eval_safety.py tests/test_acpr_flow_traffic_audit.py tests/test_acpr_flow_config_contract.py tests/test_acpr_flow_formal_path_integrations.py tests/test_acpr_flow_hardpair.py -q` 结果为 43 passed, 1 skipped。
- GitHub push：`flowtrace_pmt_v1` 已推送到 `d2116056543-lab/FATE-X`，代码 commit `7f9f8af`。

## 当前有效文件

- 远端三份记录目录：`E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree\docs\acpr_flowcalpp\`
- 本地同步目录：`E:\FATE_X_ACPR_FlowCalPP_Records\`
- 当前停止 run 有效续训 ckpt：`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030\train\checkpoint_latest.pth`

## 后续建议

1. 先从有效 latest ckpt 跑完一个完整 epoch，获得 course circular fix 后的 eval。
2. 如果 text CIDEr 继续低于历史 best，暂停增强 traffic/control 分支，回查 caption loss 权重、teacher forcing、beam/max_len、tokenizer 和 ADAPT label 对齐。
3. 对 traffic-flow 机制补 zero-out / intervention audit；目前相关性是必要证据，但还不是因果证明。
4. 继续保持每轮 eval 前保存 latest，并分开保存 best_text/best_control/best_adapt_joint。
