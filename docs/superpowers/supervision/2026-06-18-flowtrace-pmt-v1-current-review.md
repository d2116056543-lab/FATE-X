# 双代理监督审查日志：FlowTrace-PMT V1 当前落地状态

**日期：** 2026-06-18
**任务：** 严格审查 FATE-X / FlowTrace-PMT V1 是否已达到 `Codex_FATE_X_FlowTrace_PMT_V1_GoalPlan_20260618.md` 的正式完整训练前置条件。
**状态：** 需要修改
**主执行端：** 主会话
**监督端：** 同强度监督审查员

## 1. 原始请求

用户要求按 GoalPlan 完全实现 FlowTrace-PMT V1，所有权重/配置齐全，代码功能完全一致，反复确认无误后才能开启完整训练；训练必须前台监督，出现报错或数值异常及时阻止。

## 2. 适用 Skill

- `dual-agent-supervision`：用户明确要求同强度监督审查、严格计划一致性、禁止遗漏。
- `receiving-code-review`：当前是对执行证据和审查反馈进行技术判断，不能盲目批准。
- `verification-before-completion`：不能在没有 REVIEW_PASS、clean git、完整 smoke/eval 的情况下声称完成或允许训练。

## 3. 审查证据

- 远端仓库：`E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree`
- 当前 HEAD：`1d36adc214d0b4903ca725772619ccd0edd91763`
- 当前工作树：dirty，存在多处 modified/untracked 文件。
- REVIEW_PASS：`.background_runs\flowtrace_pmt_v1_preflight\REVIEW_PASS_FLOWTRACE_PMT_V1.txt` 缺失。
- 静态 audit：`.background_runs\flowtrace_pmt_v1_preflight_after_pmtalignfix\review_report.json` 的 `review_status` 为 `synthetic_dynamic_passed_formal_smoke_still_required`。
- 真实 BDD-X smoke：`.background_runs/flowtrace_pmt_v1_real_smoke_wsl_after_pmtalignfix` 已完成 1 个训练 step 并保存 `checkpoint_latest`，但 evaluation 未完整通过；命令 timeout 后进程在 evaluation 中被杀。
- 已通过测试：关键 FlowTrace 回归测试在 WSL 中通过，但这些测试不能替代完整 real-data smoke、audit pass、clean SHA 绑定。

## 4. 功能覆盖矩阵

| 编号 | 计划要求 | 必须/可选 | 当前状态 | 证据 | 审查结论 |
| --- | --- | --- | --- | --- | --- |
| 1 | 禁止正式训练，直到 REVIEW_PASS 绑定 exact clean Git commit | 必须 | 未满足 | REVIEW_PASS 缺失，工作树 dirty | 阻止正式训练 |
| 2 | 真实 direct-image smoke 必须 forward/backward、decode、intervention、visual、artifact schema 全部通过 | 必须 | 部分满足 | 仅训练 step 和 checkpoint_latest，eval 未完整结束 | 仍需补齐 |
| 3 | audit 必须写出 REVIEW_PASS | 必须 | 未满足 | audit 明确为 `formal_smoke_still_required` | 不能认可为 preflight pass |
| 4 | foreground supervisor 必须 verify review pass、clean tree、remote SHA、memory probe、attached child、epoch checkpoint | 必须 | 未验证 | 当前尚未进入正式 supervisor 运行 | 不能训练 |
| 5 | 已修复 BF16 Sinkhorn、FP16 config、PMT token alignment、gitdir mapping | 必须组件 | 可认可为局部修复 | 对应测试通过 | 认可局部修复，不等于整体通过 |
| 6 | 所有 FlowTrace losses/live decoder/intervention/visual/atlas 必须真实落地 | 必须 | 未完全验证 | 当前 evidence 只覆盖部分 tests 和 synthetic audit | 不能整体认可 |

## 5. 用户计划保真矩阵

| 编号 | 用户计划项 | 必须遵守 | 当前保留情况 | 偏离/缺口 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 1 | 完全按 GoalPlan 实现 | 必须 | 部分保留 | 尚无完整审查矩阵证明所有文件/功能/损失/工件均真实执行 | 需要修改 |
| 2 | 所有权重/配置齐全 | 必须 | 部分保留 | 资产存在但 released ADAPT checkpoint 传感器 head shape 与 2-signal 配置不一致，已用 compatible load 规避，需记录为偏差 | 需要审计说明 |
| 3 | 反复确认无误后才能训练 | 必须 | 未满足 | smoke/eval/audit/review pass 不完整 | 禁止正式训练 |
| 4 | 前台监督训练，异常及时阻止 | 必须 | 未开始 | 正式 supervisor 未通过 review pass gate | 等待 preflight 通过 |

## 6. 审查结论

**当前是否允许正式完整训练：** 否。
**最新监督状态：** changes required。

主要原因：

- REVIEW_PASS 缺失，且 GoalPlan 明确规定没有该文件禁止训练。
- 工作树 dirty，GoalPlan 要求 pass 文件绑定 exact clean Git commit，当前状态不满足。
- real-data smoke 没有完整 evaluation pass，也没有证明 decoder/intervention/visual/artifact schema 全部完成。
- audit 自身明确返回 `synthetic_dynamic_passed_formal_smoke_still_required`，不是正式批准。

## 7. 下一步要求

1. 补完整 real-data integrated smoke：训练 step、test eval、action/justification decode、state-off/random-equal-mass intervention、FlowTrace Canvas、required artifact schema、NaN/Inf 检查。
2. 修补 audit：把 formal real-data smoke gate 纳入 audit；只有所有 gates 通过才写 REVIEW_PASS。
3. 跑完整 `python -m pytest tests/test_flowtrace_*.py -q` 和必要 regression tests。
4. 提交并推送代码，确认 local HEAD 与 GitHub branch HEAD 一致。
5. 在 clean worktree 上 rerun audit，生成绑定该 exact SHA 的 `REVIEW_PASS_FLOWTRACE_PMT_V1.txt`。
6. 只有在上述全部完成后，才允许 foreground supervisor 启动 40 epoch formal train。

## 8. 最终判断

**审查状态：** changes required
**是否允许执行正式训练：** 否
**是否阻塞在用户输入：** 否，当前是工程验证缺口，需要继续补 smoke/audit/clean-SHA 流程。
