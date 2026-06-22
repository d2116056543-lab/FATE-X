# 双代理监督日志：ACPR FlowCal V2 严格补全

**日期：** 2026-06-21
**任务：** 按 ACPR FlowCal V2 计划补齐所有缺失代码细节，确保功能覆盖、配置一致、正式训练前完成严格审查。
**状态：** 执行中
**主执行端：** 主会话
**监督端：** 未创建；当前工具规则要求只有用户明确要求 sub-agent/parallel agent 时才可 spawn，本轮未显式要求。

## 1. 原始请求

用户要求“把缺少的全部补全，保证完全按照计划进行，覆盖所有代码细节和功能”。显式约束包括：

- 不只是能训练，必须功能性一致。
- 配置必须跟计划一致，不能有一点不同。
- 不得遗漏任何计划功能。
- 正式训练必须在反复确认代码层面完全落位之后再启动。
- 训练前必须保证指标、checkpoint、ADAPT 对齐、交通流机制审计等代码路径完整。

## 2. 适用 Skill

- `using-superpowers`：会话开始必须检查技能。
- `executing-plans`：用户提供了明确实现计划和文件级 checklist。
- `systematic-debugging`：远端 SSH 和 preflight 测试存在卡死/超时，需要先定位根因。
- `test-driven-development`：新增功能和修复必须先写测试约束。
- `dual-agent-supervision`：用户要求严格计划合规、无遗漏、全功能覆盖。
- `verification-before-completion`：不能在没有新鲜验证证据时声称完成。

## 3. 初始计划

1. 恢复远端 SSH，清理卡住的 pytest/python/ssh 进程。
2. 上传本地已准备的 preflight synthetic fast-path 修复。
3. 在远端 `sbw39` 跑完整 `tests/acpr_flowcal_v2`，逐个修复失败。
4. 补齐 ADAPT 合法文本评估桥，禁止从 loss 伪造 CIDEr。
5. 补齐 continuous control ADAPT 风格 RMSE/MAE/Acc@ 阈值评估。
6. 补齐五类 best checkpoint：text/explanation/control/joint/test。
7. 确保 formal training 不默认使用 `length=2` synthetic dataloader。
8. 重新生成 formal review pass，只允许 clean pushed SHA 后启动训练。

## 4. 功能覆盖矩阵

| 编号 | 用户要求/功能点 | 必须/可选 | 实现步骤 | 预期改动位置 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 完全按计划补齐代码功能 | 必须 | 对照计划和 checklist 逐项补代码 | `fate_x/acpr_flow_v2`, `fate_x/engine`, `tests/acpr_flowcal_v2` | 完整 pytest + preflight gates | 执行中 |
| 2 | 不能只做到可训练 | 必须 | 增加 formal contract tests | `tests/acpr_flowcal_v2/test_v2_formal_training_contract.py` | 测试检查 loader/checkpoint/metrics 语义 | 已本地补丁，待远端验证 |
| 3 | ADAPT 文本评估一致 | 必须 | 生成 sep-cap TSV，调用 ADAPT `two_cap_evaluate_on_coco_caption` | `adapt_caption_eval_bridge.py`, `eval_acpr_flowcal_v2.py` | 有 evalcap 时产生 CIDEr_des/CIDEr_exp；无时 blocker | 已本地补丁，待远端验证 |
| 4 | 不伪造 CIDEr | 必须 | text metrics 缺失时只输出 blocker | `eval_acpr_flowcal_v2.py` | 测试断言无 CIDEr_des/exp | 已有测试，待远端验证 |
| 5 | 控制指标按 ADAPT 连续口径 | 必须 | speed/course RMSE/MAE/Acc@ 阈值 | `eval_acpr_flowcal_v2.py` | 控制指标测试和 epoch eval 日志 | 部分已实现，待完整验证 |
| 6 | checkpoint_best_text.pth | 必须 | `BestCheckpointSuite` 按 CIDEr_des+CIDEr_exp 更新 | `train_acpr_flowcal_v2.py` | 新测试检查文件落盘 | 已本地补丁，待远端验证 |
| 7 | checkpoint_best_explanation.pth | 必须 | 按 CIDEr_exp/METEOR_exp 更新 | `train_acpr_flowcal_v2.py` | 新测试检查文件落盘 | 已本地补丁，待远端验证 |
| 8 | checkpoint_best_control.pth | 必须 | 按 speed/course RMSE 和 Acc@ 更新 | `train_acpr_flowcal_v2.py` | 新测试检查文件落盘 | 已本地补丁，待远端验证 |
| 9 | checkpoint_best_joint.pth | 必须 | 按合法 text + control 组合更新，不混离散 proxy | `train_acpr_flowcal_v2.py` | 新测试检查文件落盘 | 已本地补丁，待远端验证 |
| 10 | checkpoint_best_test.pth | 必须 | explanation-first control-safe test tuple | `train_acpr_flowcal_v2.py` | 新/旧 best selector 测试 | 已本地补丁，待远端验证 |
| 11 | formal 训练不能用 smoke/synthetic 小样本 | 必须 | 默认 `formal=True, synthetic=False`，只有显式 `--synthetic_smoke` 才小样本 | `train_acpr_flowcal_v2.py` | 新测试检查无默认 `length=2` | 已本地补丁，待远端验证 |
| 12 | preflight 不应卡在 synthetic CPU heavy path | 必须 | synthetic CPU `--allow_blocked` 走 schema-only blocked report | `run_acpr_flowcal_v2_preflight.py` | preflight JSON contract 测试 | 已本地补丁，待上传 |

## 5. 用户计划保真矩阵

| 编号 | 用户原计划项 | 必须遵循/可选 | 对应执行计划步骤 | 保留情况 | 偏离原因 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | “严格审查代码是否落地完整” | 必须遵循 | 完整 pytest + checklist + supervision log | 原样保留 | 无 | 测试和日志 | 执行中 |
| 2 | “配置完全跟计划一样” | 必须遵循 | 读取 YAML/config binding，防止默认 smoke 漂移 | 原样保留 | 无 | config binding/preflight | 待远端验证 |
| 3 | “不要有一点不同” | 必须遵循 | 对每个计划项写测试或 gate | 细化 | 需要可执行测试表达 | tests/acpr_flowcal_v2 | 执行中 |
| 4 | “再反复确认代码层面完全落位之后再启动训练” | 必须遵循 | 训练前必须 review pass + pytest + preflight | 原样保留 | 无 | 不启动训练直到完成 | 执行中 |

## 6. 监督审查

**是否已发送给监督端：** 否
**监督端 agent id：** 无
**原因：** 当前多代理工具规则要求只有用户明确要求 sub-agent/parallel agent work 才可 spawn；本轮用户要求严格审查但未显式要求创建子代理，因此不创建子代理。

**主会话自检结论：**

- 必须补齐 ADAPT sep-cap 文本评估桥，否则 text/joint/test best 不能合法更新。
- 必须移除 formal suite 默认 `length=2`，否则训练数据与计划不一致。
- 必须把五类 best checkpoint 独立更新，不能只存 `checkpoint_best_test.pth`。
- 必须保持 CIDEr blocker 机制，不能用 loss 或空文本伪造指标。

## 7. 计划修订

已采纳：

- 新增 `BestCheckpointSuite`。
- 新增 `adapt_caption_eval_bridge.py`。
- 增加 `generate_adapt_caption_pairs`，用模型 logits 生成 description/explanation pair，不回填 ground truth。
- `run_formal_suite` 默认 formal dataloader，不再默认 `length=2`。

未采纳及理由：

- 暂未创建子代理：工具规则限制。
- 暂未启动训练：远端 SSH 不稳定，且完整验证尚未完成。

## 8. 复审轮次

| 轮次 | 修订内容 | 结论 | 是否允许执行训练 | 剩余问题 |
| --- | --- | --- | --- | --- |
| 1 | 本地补齐 checkpoint/eval/formal loader 合同 | 需要远端验证 | 否 | 远端 SSH 卡在连接/握手，无法上传和跑 sbw39 pytest |

## 9. 执行交接

**是否已发给执行端：** 是，主会话继续执行。

**交接内容：**

- 等远端 SSH 恢复后上传本地 patch 区文件。
- 先跑新增 targeted tests，再跑完整 `tests/acpr_flowcal_v2 -vv -x`。
- 若测试失败，按 systematic-debugging 定位根因，不启动训练。

## 10. 执行合规检查

**执行端是否照做：** 部分
**所有必须功能是否完整实现：** 部分
**所有必须计划项是否保留：** 是，但尚未全部验证

**证据：**

- 本地 `python -m py_compile ...` 通过。
- 本地静态检索确认 `length=2` formal 默认已移除。
- 新增 test 文件覆盖 formal loader 与 best checkpoint。

**偏离：**

- 远端不可访问导致无法立即上传、运行远端测试、生成正式 review pass。

## 11. 验证证据

**已运行命令：**

- `python -m py_compile ...`
- `rg -n "length=2|checkpoint_best_text|checkpoint_best_explanation|checkpoint_best_control|checkpoint_best_joint|checkpoint_best_test|CIDEr|text_metrics_blocker|synthetic_smoke|formal" ...`
- `Test-NetConnection 100.75.8.120 -Port 22`
- `ssh -o ConnectTimeout=8 ...`

**结果摘要：**

- 语法检查通过。
- TCP 22 端口检测可通，但 SSH 命令仍超时/卡住。
- 本地环境无 torch，无法替代远端 sbw39 做 pytest。

## 12. 最终判断

**是否可以报告完成：** 否

**理由：**

- 补丁已在本地 patch 区完成一部分，但远端还未上传。
- 完整 pytest、formal preflight、正式 review pass 尚未完成。
- 训练尚未启动，也不应在验证完成前启动。
