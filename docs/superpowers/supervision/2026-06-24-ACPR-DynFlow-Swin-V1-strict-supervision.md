# 双代理监督日志：ACPR-DynFlow-Swin V1 严格实现

**日期：** 2026-06-24
**任务：** 根据用户提供的 ACPR-DynFlow-Swin V1 计划、配置、manifest、file-level checklist 和 audit skill，在当前 `acpr_dynflow_v1` branch 上完整实现代码功能，不遗漏计划项。
**状态：** 规划与合同安装中
**主执行端：** 主会话 Codex
**监督端：** 工具限制：当前多代理工具仅允许在用户明确要求 sub-agent/delegation 时 spawn；本任务用户要求“审查机制”但未显式要求创建子代理，因此不启动 sub-agent。按 dual-agent-supervision 规则记录限制，并由主会话执行逐项覆盖矩阵与保真矩阵。

## 1. 原始请求

用户要求：当前代码有很多问题，按照发送的 DynFlow-Swin V1 指导文件严格修改，保证代码功能全部落位，不要遗漏；如果不能完成就持续做下去，把整套代码搞好。

显式约束：

- 不新建 worktree，不新建 branch。
- 当前 worktree：`E:\sbw\FATE_Drive\fate_x_acpr_dynflow_v1_worktree`。
- 当前 branch：`acpr_dynflow_v1`。
- 计划、checklist、config、manifest、audit skill 是硬合同。
- 正式训练前必须通过 preflight/review pass；不允许只做能训练的表面实现。

## 2. 适用 Skill

- `using-superpowers`：会话开始必须检查并使用相关 skill。
- `dual-agent-supervision`：用户要求严格计划合规、无遗漏和审查机制。
- `executing-plans`：用户提供了完整 implementation plan。
- `test-driven-development`：本任务涉及新功能和 bug fix，需测试先行。
- `verification-before-completion`：任何完成声明前必须有新鲜验证证据。
- `requesting-code-review`：重大实现完成后需审查；当前无显式 sub-agent 授权，先用静态/动态审查命令替代并记录限制。

## 3. 初始计划

1. 安装合同文件到仓库指定路径。
2. 完整读取 canonical ledgers、runbook、manifest、skill、config，记录哈希。
3. 建立覆盖矩阵和保真矩阵。
4. 先写测试，暴露当前 formal namespace 缺失与旧缺陷。
5. 独立实现 `fate_x.acpr_dynflow_swin`，禁止 formal import 旧 DynFlow/FlowCal/FlowTrace。
6. 实现 preflight/audit/supervisor 入口，使训练授权受 review pass 约束。
7. 跑 compile/test/import/preflight smoke，失败则继续修。
8. 提交、推送、验证 GitHub SHA 相等。

## 4. 功能覆盖矩阵

| 编号 | 用户要求/功能点 | 必须/可选 | 实现位置 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | 安装计划、配置、manifest、audit skill | 必须 | `docs/runbooks`, `configs`, `.codex/skills` | 文件存在与 SHA 记录 | 已实施，待提交 |
| 2 | 新 formal namespace `fate_x.acpr_dynflow_swin` | 必须 | `fate_x/acpr_dynflow_swin/*` | import graph + tests | 未开始 |
| 3 | 禁止 legacy formal imports | 必须 | audit + tests | AST/runtime import scan | 未开始 |
| 4 | Direct image `[B,32,3,224,224]` | 必须 | data/model/backbone | smoke + no-cache audit | 未开始 |
| 5 | Repo Video Swin-B native stages, one forward, BF16 | 必须 | `video_swin_backbone.py`, shared Swin | backbone tests/audit | 未开始 |
| 6 | 32 OIA predicate transfer | 必须 | ontology/transfer/field | ontology + query tests | 未开始 |
| 7 | nnPU/CalAlign real labels | 必须 | `nnpu_calalign.py` | synthetic text tests | 未开始 |
| 8 | Mass-preserving consolidation | 必须 | consolidator | identity tests | 未开始 |
| 9 | Pattern-lag traffic reasoner | 必须 | traffic reasoner | synthetic pattern/lag tests | 未开始 |
| 10 | Motion transformer + exact ledger | 必须 | motion/ledger/losses | target independence + ledger identity | 未开始 |
| 11 | ADAPT-compatible autoregressive text | 必须 | text decoder/eval | decode/loss split tests | 未开始 |
| 12 | Per-epoch test eval/checkpoints | 必须 | trainer/evaluator | best selector/checkpoint tests | 未开始 |
| 13 | Real interventions/visuals/atlas | 必须 | interventions/explain | schema + recompute tests | 未开始 |
| 14 | Throughput/preflight/review pass | 必须 | probe/preflight/audit/supervisor | gate reports + pass file | 未开始 |

## 5. 用户计划保真矩阵

| 编号 | 用户原计划项 | 必须遵守/可选 | 对应执行计划 | 保留情况 | 验证方法 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 不新建 worktree/branch | 必须 | 在当前 worktree/branch 实施 | 原样保留 | git branch/status | 已验证 |
| 2 | formal namespace 独立，不导入旧实现 | 必须 | 新建 `fate_x.acpr_dynflow_swin` | 原样保留 | import graph test | 未开始 |
| 3 | 训练前 review pass | 必须 | preflight/audit 写 pass | 原样保留 | pass 文件绑定 SHA | 未开始 |
| 4 | 16 epoch 固定训练，不 stage 切换 | 必须 | trainer config 固定 | 原样保留 | trainer tests | 未开始 |
| 5 | projected epoch >4h 阻止 formal launch | 必须 | throughput probe gate | 原样保留 | probe report | 未开始 |
| 6 | 前台 supervisor 不 detach | 必须 | foreground scripts/supervisor | 原样保留 | static scan + smoke | 未开始 |

## 6. 监督审查

**是否已发送给独立监督端：** 否，工具规则限制未获显式 sub-agent 授权。
**审查方式：** 主会话逐项矩阵审查，后续用 audit skill 和测试命令提供动态证据。
**当前结论：** 不允许进入正式训练；当前只允许合同安装、代码实现、测试和 preflight 修复。

## 7. 执行记录

- 已验证 worktree：`E:\sbw\FATE_Drive\fate_x_acpr_dynflow_v1_worktree`。
- 已验证 branch：`acpr_dynflow_v1`。
- 已验证初始 HEAD 与 GitHub：`7da4f84d40beb6908b1f34f0b134709317d5a0e4`。
- 已发现合同文件在仓库中缺失，已安装到指定路径。
- 已完整读取十个 required context 文件，并写入 `.background_runs/acpr_dynflow_swin_v1_context_read.json`。

## 8. 待验证证据

- 合同文件 commit/push/SHA equality。
- 后续每批实现的 targeted tests、compileall、git diff --check。
- 最终 preflight gate reports 和 review pass。
