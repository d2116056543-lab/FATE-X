# 双代理监督日志：ACPR DynFlow V1 严格实现与启动核验

**日期：** 2026-06-23
**任务：** 根据 ACPR DynFlow V1 计划和用户提供文件，严格实现所有代码功能，确保代码真实调用，不假装完成，并启动正式实验直到输出 batch。
**状态：** 已执行并完成启动核验
**主执行端：** 当前 Codex 会话
**监督端：** 主会话按 dual-agent-supervision 覆盖矩阵自审；未另启低强度监督代理

## 1. 原始请求

用户要求根据 `acpr_dynflow_v1_bddx_32f_224.yaml`、文件级 checklist、package manifest、implementation audit skill、bootstrap prompt 和 implementation plan 严格执行；必须 100% 覆盖计划，不允许假实现、未调用或只写外壳；实验必须开始且没有报错并输出 batch。

## 2. 适用 Skill

- `dual-agent-supervision`：用户明确要求严格计划一致、完整覆盖、不能遗漏。
- `verification-before-completion`：完成前必须用真实命令和日志证明代码、测试、push、smoke 和正式 batch。

## 3. 功能覆盖矩阵

| 编号 | 用户要求/功能点 | 必须/可选 | 实现或核验位置 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | 按 DynFlow V1 计划和包文件执行 | 必须 | `configs/acpr_dynflow_v1_bddx_32f_224.yaml`、`fate_x/engine/train_acpr_dynflow.py`、`fate_x/*dynflow*`、`tests/acpr_dynflow` | compileall、pytest、review report | 已验证 |
| 2 | 不假装完成，代码必须真实调用 | 必须 | trainer 输出 `ACPR_DYNFLOW_BATCH`，loss components JSONL | smoke 与正式 run 日志 | 已验证 |
| 3 | 评估 sample cap 不能漂移 | 必须 | `evaluation.best_checkpoint_cases=256`，trainer fallback 链 | `training_effective_batch.json` 显示 `eval_max_samples=256` | 已验证 |
| 4 | GitHub branch 同步 | 必须 | branch `acpr_dynflow_v1` | `git ls-remote` 与 HEAD 一致 | 已验证 |
| 5 | 正式实验后台不因 SSH 断开停止 | 必须 | Windows Task Scheduler `.cmd` launcher | `schtasks` 与 python process active | 已验证 |
| 6 | 输出 batch 后才算启动成功 | 必须 | `train.log` / `loss_components.jsonl` | 正式 run first batch line | 已验证 |

## 4. 用户计划保真矩阵

| 编号 | 用户原计划项 | 必须遵循 | 保留情况 | 证据 |
| --- | --- | --- | --- | --- |
| 1 | 严格按计划和发送文件实现 | 必须 | 保留 | 53 个 ACPR DynFlow 测试通过，review pass 生成 |
| 2 | 不要漏任何细节 | 必须 | 保留到可验证范围 | checklist/test/review report 覆盖；发现 eval cap 漂移后已修复并重新验证 |
| 3 | 不能假装完成实际没实现或没调用 | 必须 | 保留 | smoke 与正式 run 均产生真实 batch 和 loss components |
| 4 | 实验开始且无报错并输出 batch | 必须 | 保留 | 正式 run `ACPR_DYNFLOW_BATCH` 已输出 |

## 5. 监督审查结论

审查发现一个关键漂移：正式训练原先没有读取 `visualization.best_checkpoint_cases=256`，导致正式 run 的 `eval_max_samples=-1`。执行侧已停止该无效正式 run，修复 trainer 和 config，增加测试，并重新验证后重新启动正式 run。该项已闭环验证。

## 6. 计划修订和纠正动作

- 新增 `evaluation.best_checkpoint_cases: 256`。
- 修改 `train_acpr_dynflow.py` 的 eval cap fallback：CLI override > `evaluation.best_checkpoint_cases` > `visualization.best_checkpoint_cases` > `evaluation.lightweight_flow_audit_samples` > -1。
- 更新 `tests/acpr_dynflow/test_best_selectors.py` 覆盖新的 fallback。
- 重新运行 compile、targeted test、full test、diff check、smoke、正式 batch 核验。

## 7. 验证证据

- `python -m compileall fate_x tests\acpr_dynflow -q`：exit 0。
- `pytest tests\acpr_dynflow\test_best_selectors.py -q`：2 passed。
- `pytest tests\acpr_dynflow -q`：53 passed。
- `git diff --check`：exit 0。
- GitHub branch `acpr_dynflow_v1`：remote HEAD = `50505c1fc3c928a60a06b65b58b53811dcc04266`。
- Smoke run：`G:\sbw\FATE_Drive\active_runs\acpr_dynflow_v1_smoke_50505c1_cmd_20260623_084919`，exit 0，输出 batch 和 ADAPT 文本评估链。
- 正式 run：`G:\sbw\FATE_Drive\active_runs\acpr_dynflow_v1_formal_50505c1_schtasks_20260623_085221`。
- 正式 batch：epoch 0 batch 0 global_step 1，loss `19.649354934692383`，`frames_shape=[10,32,3,224,224]`，`gradient_accumulation_steps=7`，`optimizer_stepped=false`。
- 正式 effective batch：batch size 10，grad accumulation 7，effective batch size 70，optimizer steps per epoch 235，`eval_max_samples=256`。
- GPU：batch 后约 39460MiB / 49140MiB。

## 8. 当前风险

- TensorFlow 的 `cudart64_110.dll not found` 警告来自 eval/tokenizer 相关路径；smoke 和正式 batch 均已继续执行，目前不是阻塞。
- 正式训练后续 epoch 指标仍需持续监控，当前日志只证明正式 run 已进入训练 batch 且配置关键项正确。

## 9. 最终判断

可以报告“代码已推送、关键验证通过、正式实验已启动并输出 batch”。不能报告“完整训练已完成”或“最终指标达标”，因为正式训练仍在运行。


## 10. 2026-06-23 二次监督更新：masked text supervision 修复与 e66bce9 正式 run

**状态：** 已发现并修复一项会使正式训练无效的关键监督错误；修复后正式训练已重新启动并输出有效 batch。

### 10.1 发现的问题

- 50505c1 正式 run 虽然能输出 batch，但 `loss_components.explanation_text = 0.0`。
- 这与配置 `loss.explanation_text = 0.50` 冲突，表示 explanation 半段文本没有被监督，不能继续当作有效正式实验。
- 已停止无效任务 `acpr_dynflow_v1_full_50505c1_20260623_085221`。

### 10.2 根因

- BDD-X/ADAPT dataloader 的 `masked_pos` 是 `[B, 30]` 二值 mask，不是显式 token position list。
- 旧 `DynFlowTextDecoder._masked_position_loss` 把 0/1 mask 值当作位置索引，只监督 token 0/1，导致 explanation half 没有有效目标。

### 10.3 修复内容

- 文件：`fate_x/acpr_dynflow/text_decoder.py`
- 修复：识别二值 mask 格式，将 `masked_pos` 转换为真实 token positions，并用 packed `masked_ids` 标签计算 action/explanation 两段 CE。
- 新增回归测试：`tests/acpr_dynflow/test_text_decoder_masked_positions.py`

### 10.4 验证证据

- `pytest tests\acpr_dynflow\test_text_decoder_masked_positions.py -q`：1 passed。
- `python -m compileall fate_x tests\acpr_dynflow -q`：exit 0。
- `pytest tests\acpr_dynflow -q`：54 passed。
- `git diff --check`：exit 0。
- 修复提交：`e66bce98285883315b949250dd73ec17df3f3214`，已 push 到 `github/acpr_dynflow_v1`。
- preflight：`.background_runs\acpr_dynflow_v1_final_preflight_20260623_e66bce9`，`passed=true`。

### 10.5 修复后 smoke

- run：`G:\sbw\FATE_Drive\active_runs\acpr_dynflow_v1_smoke_e66bce9_maskfix_20260623_0919`
- smoke exit：0。
- batch 0 loss：`28.541072845458984`。
- `action_text = 10.429052352905273`。
- `explanation_text = 10.449007987976074`。
- traffic prediction correlation 非空：`pred_speed_delta_corr=-0.15232357382774353`，`pred_course_delta_corr=-0.3011300563812256`。

### 10.6 修复后正式 run 当前状态

- task：`acpr_dynflow_v1_full_e66bce9_20260623_0922`
- run：`G:\sbw\FATE_Drive\active_runs\acpr_dynflow_v1_formal_e66bce9_maskfix_schtasks_20260623_0922`
- 训练命令：`E:\Anaconda\envs\sbw39\python.exe -u -m fate_x.engine.train_acpr_dynflow --config configs\acpr_dynflow_v1_bddx_32f_224.yaml --output_dir <run> --device cuda`
- 当前确认 batch：`249/1639` micro-batches，epoch 0 约 15.2%。
- 有效 batch：`batch_size=10`，`gradient_accumulation_steps=7`，`effective_batch_size=70`，`optimizer_steps_per_epoch=235`，`eval_max_samples=256`。
- GPU：约 `45564 MiB / 49140 MiB`。
- 当前 loss 未见 NaN/Inf；`action_text` 和 `explanation_text` 均持续非零并下降。
- 重要更正：`optimizer_steps_per_epoch=235` 不是 epoch 的 micro-batch 数；真实每 epoch 需要约 `1639` 个 micro-batch，因此单卡完整 epoch 很慢。当前 run 仍在训练中，尚未到 epoch 0 checkpoint/eval。

### 10.7 当前监督结论

- 可以确认：代码已修复上一版无效 explanation 监督问题，正式 run 已输出有效 batch，GPU/有效 batch/文本监督均符合当前配置。
- 不能确认：epoch 0 的 checkpoint/eval/traffic-flow audit 已完成；因为当前尚未到 epoch 末尾。
- 后续必须继续检查：`checkpoint_latest.pth`、`checkpoint_best_text.pth`、`checkpoint_best_control.pth`、`metrics_summary.jsonl`、`traffic_flow_audit.pred_speed_delta_corr` 与 `pred_course_delta_corr`。
