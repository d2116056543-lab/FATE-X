<!--
Canonical ACPR FlowCalPP / FlowCal V2 Findings ledger.
This file intentionally contains both the earlier V1/FlowCalPP record and the later V2 record.
Restored and merged from git commit ccb0370 on 2026-06-23 00:32:59 Asia/Shanghai after the user requested one continuous three-file history.
Do not split V2 into separate task/findings/progress files again.
-->

# Unified ACPR FlowCalPP / FlowCal V2 Findings Ledger

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

# ACPR FlowCalPP Findings

更新时间：2026-06-21 14:25 Asia/Shanghai

## 1. 总结论

当前最可靠结论分三层：

1. 训练已经停止在 `acpr_linux_b4w4_coursecircularfix3_20260621_121030` 的 epoch 15 中途，该停止点没有 test eval，不能作为最终结果。
2. 严格/current mechanism 的最新完整评价是 epoch14：control 接近 ADAPT，文本明显落后。
3. historical text best 的文本更好，但属于旧诊断口径时期，不能直接代表当前 traffic-flow/circular-control 完整机制。

最重要的问题不是“有没有训练起来”，而是：

- 文本生成没有达到 ADAPT 水平。
- 交通流机制目前有相关性证据，但还缺 zero-out/counterfactual 因果证据。
- course circular fix 已进入代码，但被用户要求停止的 run 没有完成新 eval。
- 一天多运行中暴露出多个工程问题，必须保留为后续防复发经验。

## 2. ADAPT 对比

ADAPT 原文常见表格数值按 x100 展示；这里换成 eval JSON 小数口径。

| Item | ADAPT paper approx | strict epoch14 | historical text best |
|---|---:|---:|---:|
| CIDEr_des | 2.4750 | 0.6488 | 1.4878 |
| CIDEr_exp | 1.0260 | 0.5441 | 0.7872 |
| CIDEr_des+exp | 3.5010 | 1.1930 | 2.2750 |
| Speed RMSE | 2.5000 | 2.5398 | 2.6372 |
| Speed Acc@0.5 | 0.2810 | 0.2894 | 0.2482 |
| Speed Acc@1 | 0.4530 | 0.4492 | 0.4247 |
| Course RMSE | 6.4000 | 6.1176 | 6.1163 |
| Course Acc@0.5 | 0.8550 | 0.8676 | 0.8681 |
| Course Acc@1 | 0.8990 | 0.9102 | 0.9107 |

解释：

- Control：strict epoch14 的 speed RMSE 与 ADAPT 接近，speed Acc@0.5 略高，speed Acc@1 基本持平；course RMSE 和 threshold accuracy 也不差。
- Text：strict epoch14 的 CIDEr_des+exp 明显低，historical text best 仍低于 ADAPT，但更接近。
- 因此当前主要短板不是 control regression，而是 caption/text generation 与机制分支的 tradeoff。

## 3. Strict epoch14 详细指标

文件：

`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_resume_envfix_20260621_065540\train\eval_epoch_014.json`

文本：

| Metric | description | explanation |
|---|---:|---:|
| Bleu_1 | 0.3195 | 0.1394 |
| Bleu_2 | 0.2777 | 0.0769 |
| Bleu_3 | 0.1701 | 0.0515 |
| Bleu_4 | 0.1178 | 0.0370 |
| CIDEr | 0.6488 | 0.5441 |
| METEOR | 0.2009 | 0.0981 |
| ROUGE_L | 0.5046 | 0.2248 |
| SPICE | 0.5468 | 0.1534 |

Control：

| Signal | RMSE | MAE | Acc@0.1 | Acc@0.5 | Acc@1 | Acc@5 | Acc@10 | valid_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| speed | 2.5398 | 1.7603 | 0.0691 | 0.2894 | 0.4492 | 0.9406 | 0.9958 | 67729 |
| course | 6.1176 | 0.8891 | 0.5514 | 0.8676 | 0.9102 | 0.9734 | 0.9878 | 67936 |

Selection scores：

- `text_cider = 1.1929573563`
- `control_rmse_negative = -8.6574087143`
- `control_threshold_mean = 0.6290806234`
- `adapt_joint = 0.9562971083`

## 4. Historical text best 详细指标

文件：

`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_resume_epoch5_foreground_20260620_185230\train\eval_epoch_005.json`

文本：

| Metric | description | explanation |
|---|---:|---:|
| Bleu_1 | 0.4061 | 0.2054 |
| Bleu_2 | 0.3682 | 0.1262 |
| Bleu_3 | 0.2966 | 0.0886 |
| Bleu_4 | 0.2448 | 0.0667 |
| CIDEr | 1.4878 | 0.7872 |
| METEOR | 0.2352 | 0.1229 |
| ROUGE_L | 0.5610 | 0.2844 |
| SPICE | 0.5710 | 0.2199 |

Control：

| Signal | RMSE | MAE | Acc@0.1 | Acc@0.5 | Acc@1 | Acc@5 | Acc@10 | valid_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| speed | 2.6372 | 1.8662 | 0.0543 | 0.2482 | 0.4247 | 0.9320 | 0.9955 | 67729 |
| course | 6.1163 | 0.8804 | 0.5895 | 0.8681 | 0.9107 | 0.9733 | 0.9878 | 67936 |

为什么不能直接当最终结果：

- 该时期 eval 里仍有旧离散 decision proxy 字段。
- 它可以说明文本分支有潜力，但不能说明当前 strict traffic-flow/circular-control 机制也有同等文本表现。

## 5. 交通流审计结论

strict epoch14 已经输出 target control delta 和 predicted control delta 的相关性。

| Factor | mean | std | target_speed_corr | pred_speed_corr | target_course_corr | pred_course_corr |
|---|---:|---:|---:|---:|---:|---:|
| queue_congestion | 0.3902 | 0.3303 | -0.1941 | -0.6591 | -0.0148 | -0.5277 |
| clear_open_flow | 0.2001 | 0.2127 | 0.2832 | 0.4172 | 0.0185 | 0.5259 |
| traffic_signal | 0.2523 | 0.2375 | -0.1414 | -0.5146 | -0.0060 | -0.2787 |
| turn_intersection | 0.3110 | 0.1827 | 0.1689 | -0.2454 | 0.0046 | 0.2657 |
| lead_vehicle_group | 0.1745 | 0.1857 | 0.0042 | -0.0763 | 0.0114 | -0.2505 |
| dense_following | 0.0196 | 0.0218 | 0.0285 | 0.6360 | 0.0089 | 0.2369 |
| forming | 0.0219 | 0.0302 | 0.0199 | 0.5855 | 0.0030 | 0.1935 |

可解释结论：

- `queue_congestion` 和 `traffic_signal` 与真实 speed delta 负相关，模型预测 speed delta 也强负相关，说明模型输出已经对拥堵/信号相关因子有反应。
- `clear_open_flow` 与真实 speed delta 正相关，模型预测 speed/course delta 也正相关，说明开阔交通流对模型输出有正向影响。
- `dense_following`、`forming` 的 target 相关较弱但 pred 相关较强，可能说明模型对这些因子过敏，需用 zero-out 验证是否是误用。
- `lead_vehicle_group` 的 target 相关弱、pred 相关为负，可能是模型把前车组当作减速提示，但这仍只是相关性，不能直接说是因果机制。

当前不足：

- 还没有 zero-out / intervention 证据。
- course 相关性来自 course circular fix 前，需新 eval 验证。
- 还未证明 traffic factor 改善了文本 CIDEr。

## 6. 组件作用判断

| Component | 当前证据 | 判断 | 下一步 |
|---|---|---|---|
| ADAPT-aligned text eval | des/exp COCO metrics 正常输出 | 已落地 | 回查文本低分原因 |
| Continuous control eval | speed/course RMSE 和 threshold accuracy 正常输出 | 已落地 | 保持为主 control 指标 |
| Checkpoint split | latest/best_text/best_control/best_adapt_joint/best_test 存在 | 已落地 | 后续继续分开选模 |
| Traffic-flow audit | target 和 pred delta corr 均非 null | 已落地 | 加 zero-out/counterfactual |
| Flow PU | batch loss 中持续有 `flow_pu` | 参与训练 | 需证明贡献文本或 control |
| Predicate PU | batch loss 中持续有 `predicate_pu` | 参与训练 | 需审计 predicate 质量 |
| Reason semantic | batch loss 中有 `reason_semantic` | 参与训练 | 未证明提升 CIDEr |
| Hardpair | 最后 batch active pair rate 1.0，测试修复后投影有梯度 | 已启用 | 观察对文本 tradeoff |
| Future control | batch loss 中有 `future_control`，但有 spike | 已启用 | 需要稳定性审计 |
| Course circular fix | 代码已改，run 未完成 eval | 未完成验证 | 续训一轮获得 eval |

## 7. 错误经验库

| # | 问题 | 触发条件/现象 | 根因 | 修复 | 验证/证据 | 防复发 |
|---:|---|---|---|---|---|---|
| 1 | 只看训练能跑，没看 eval | 早期 smoke 能训练，但不知道是否和 ADAPT 对齐 | smoke 只验证 forward/backward，不验证 metric contract | 加 per-epoch test eval | 产生 `eval_epoch_XXX.json` | 任何完整训练必须包含 eval 输出 |
| 2 | 文本有 CIDEr，control 没合法指标 | eval 只有 CIDEr_des/exp | 之前只按 ADAPT caption 做输出 | 加 continuous control eval | speed/course RMSE、MAE、Acc@threshold 正常输出 | 文本和 control 分开检查 |
| 3 | 离散 decision proxy 坍缩 | speed 只预测 maintain，course 只预测 straight，macro recall 0.3333 | BDD-X 原始是连续 control，不适合硬转离散主指标 | 降级为历史诊断，不再主评估 | strict eval 不再依赖该 proxy | 不再把 proxy 混入 best 选择 |
| 4 | eval 出错可能浪费整轮 | eval 在保存 ckpt 前运行 | 一轮训练成果依赖 eval 成功 | eval 前先保存 latest | latest ckpt 独立存在 | 保持先存后评 |
| 5 | best ckpt 单一 | text/control/joint 目标冲突 | 一个 best 无法表达多目标 | 拆成 best_text/control/adapt_joint/test | strict run 有多个 best 文件 | 后续不合并 |
| 6 | traffic audit 只能看标签相关 | `pred_speed_delta_corr`/`pred_course_delta_corr` 为 null | 只统计 target delta，没有统计 prediction delta | 加 prediction delta audit | epoch14 pred corr 有数值 | 交通流审计必须 target/pred 双通道 |
| 7 | course 角度 wrap-around | course delta 可能出现假大误差 | 角度是环形变量，不能纯线性相减 | 加 circular course delta fix | 代码已改，但需要新 eval | 不能用未验证 run 宣称效果 |
| 8 | hardpair 训练路径未必有梯度 | 测试发现 `model.hardpair.proj.weight.grad is None` | 无 BERT embedding fallback 用 predicted state，导致无 eligible pair | fallback 改为使用 `reason_for_pair` target | 单项测试和全套 ACPR 测试通过 | formal-path 测试必须覆盖梯度 |
| 9 | 训练中途停止留下 partial ckpt | 停止时出现 `checkpoint_latest.pth.tmp` | 正在写 checkpoint 时被停止 | 标记 `.tmp` 无效，只用完整 latest | 文件大小和 mtime 已记录 | resume 前检查 tmp |
| 10 | SSH PowerShell 命令跑错目录 | `Set-Location` 没生效，git 报 not repo | 嵌套引号/一行命令在远端解释失败 | 改用 EncodedCommand / 临时脚本 | 后续验证改用脚本方式 | 不用复杂 one-liner 做关键验证 |
| 11 | WSL env assignment 被 Windows 解释 | `CUDA_VISIBLE_DEVICES` 被报不是命令 | 直接在 Windows shell 中写 bash env assignment | 用 `wsl -- bash -lc` | 测试命令可运行 | 所有 Linux 命令显式进入 WSL |
| 12 | CRLF/尾随空白污染 | `git diff --check` 报大量 trailing whitespace | Windows/补丁混合写入 | 统一 LF，清理尾随空白 | `git diff --check` 通过 | commit 前必跑 diff check |
| 13 | 把 old best 和 strict best 混读 | historical text best 文本更高 | 不同机制/口径阶段结果混在一起 | 文档分层记录 | 本文明确分 strict 和 historical | 对比时必须写 run 和 eval 文件 |

## 8. 最近 stopped run 的 batch 级观察

Run：

`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_coursecircularfix3_20260621_121030\train`

最后 batch：

- epoch：`15`
- global_step：`64500`
- optimizer_step：`3717`
- loss：`0.7540768981`
- precision：`bf16`
- sample ids：`training_14eb5976-0844b87f_14125:0`、`training_03f6e688-415b77da_01785:0`

loss components：

| Component | Raw | Weighted |
|---|---:|---:|
| action_text | 0.1764 | 0.1764 |
| explanation_text | 0.4486 | 0.4486 |
| control | 1.1019 | 0.0551 |
| predicate_pu | 0.4291 | 0.0215 |
| flow_pu | 0.2880 | 0.0086 |
| reason_semantic | 0.5427 | 0.0271 |
| future_control | 0.7403 | 0.0148 |
| memory_diversity | 0.5324 | 0.0005 |
| hardpair_raw_loss | 0.0491 | 0.0015 |

解释：

- 该 batch 里 hardpair 已 active，candidate_count 为 `4096`。
- 控制项 raw loss 不低，但 weighted 后占比有限。
- caption text loss 仍是主要贡献之一，后续如果 CIDEr 不涨，需要重点看文本分支学习能力，而不是只看总 loss。

## 9. 当前最大技术风险

1. Text CIDEr 下滑风险：strict epoch14 明显低于 historical text best，可能是机制分支/权重影响文本生成，也可能是 run 差异。
2. Course circular fix 未验证：代码修复后没有完整 eval，不能判断是否有效。
3. Traffic factor 仍缺因果证据：相关性存在，但需要 zero-out/counterfactual。
4. Hardpair 可能带来 tradeoff：formal path 已修，但对 CIDEr/control 的长期影响仍要看曲线。
5. Future control loss spike：batch tail 中 loss 有高波动，需监控是否影响文本。
6. 远端/WSL/SSH 操作容易因 shell 差异导致错误路径、错误环境或错误进程。

## 10. 下一轮必须回答的问题

- course circular fix 后，course RMSE / threshold accuracy 是否变化？
- strict 机制继续训练后，CIDEr 是回升到 historical text best 附近，还是继续被压低？
- traffic factor zero-out 是否真的改变模型 control prediction？
- hardpair active 后，description/explanation CIDEr 是提升还是下降？
- control 与 text 是否出现明确 tradeoff？
- 当前 loss 权重是否需要改成 staged schedule，而不是所有机制一起训练？


---

# Part B: Detailed V2 Record Appended To The Same Ledger

# ACPR FlowCal V2 Findings

Generated: 2026-06-23 00:12:56

## Bottom Line

当前 ACPR FlowCal V2 run 没有超过 ADAPT 复现第 4 轮，也没有接近 ADAPT 复现 best 或 ADAPT 原文。按真实远端 eval 文件计算：

| Run | CIDEr_des | CIDEr_exp | CIDEr_sum | CIDEr_sum x100 | speed RMSE | course RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| ADAPT repro epoch4 | 2.3225 | 0.8410 | 3.1634 | 316.3 | 2.9790 | 6.1208 |
| ADAPT repro best epoch12 | 2.3883 | 0.9107 | 3.2989 | 329.9 | 2.3625 | 6.1193 |
| ACPR FlowCal V2 best epoch2 | 1.3208 | 0.7553 | 2.0761 | 207.6 | 7.0209 | 88.9550 |
| ACPR FlowCal V2 latest epoch4 | 1.3210 | 0.7474 | 2.0685 | 206.8 | 6.9460 | 88.9634 |

## Direct Answer

- 是否超过 ADAPT 复现 epoch 4：没有。
- V2 best `CIDEr_des+exp=2.0761`，ADAPT 复现 epoch4 `CIDEr_des+exp=3.1634`。
- 差距：`-1.0874` raw CIDEr，论文百分制约 `-108.7`。
- 是否超过 ADAPT 复现 best：没有。ADAPT 复现 best 是 epoch 12，`CIDEr_des+exp=3.2989`。

## Current V2 Eval History

| Epoch | Stage | CIDEr_des | CIDEr_exp | CIDEr_sum | speed RMSE | course RMSE | speed A0.5 | course A0.5 | pred speed corr | pred course corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | semantic_recovery | 1.3204 | 0.7272 | 2.0476 | 7.0222 | 88.9551 | 0.1663 | 0.2628 | -0.1775 | -0.1712 |
| 2 | semantic_recovery | 1.3208 | 0.7553 | 2.0761 | 7.0209 | 88.9550 | 0.1665 | 0.2628 | -0.1539 | -0.1470 |
| 4 | axis_aware_motion | 1.3210 | 0.7474 | 2.0685 | 6.9460 | 88.9634 | 0.1766 | 0.2628 | -0.1297 | -0.1221 |

## ADAPT Reproduction Eval History

| Epoch | Checkpoint | CIDEr_des | CIDEr_exp | CIDEr_sum | speed RMSE | course RMSE | speed A0.5 | course A0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | checkpoint-0-1 | 0.0000 | 0.0000 | 0.0000 | 8.7613 | 6.1734 | 0.0494 | 0.2997 |
| 1 | checkpoint-1-256 | 1.3969 | 0.4351 | 1.8320 | 3.9263 | 6.1195 | 0.0993 | 0.8536 |
| 2 | checkpoint-2-512 | 1.6973 | 0.6164 | 2.3137 | 2.8685 | 6.1231 | 0.1203 | 0.8552 |
| 3 | checkpoint-3-768 | 1.7595 | 0.8077 | 2.5672 | 3.1185 | 6.1166 | 0.2111 | 0.8699 |
| 4 | checkpoint-4-1024 | 2.3225 | 0.8410 | 3.1634 | 2.9790 | 6.1208 | 0.2621 | 0.8495 |
| 5 | checkpoint-5-1280 | 2.0468 | 0.7330 | 2.7797 | 2.8960 | 6.1194 | 0.0992 | 0.8568 |
| 6 | checkpoint-6-1536 | 1.8826 | 0.8137 | 2.6963 | 2.4663 | 6.1175 | 0.3003 | 0.8650 |
| 7 | checkpoint-7-1792 | 2.1015 | 0.7597 | 2.8612 | 2.4233 | 6.1177 | 0.3277 | 0.8600 |
| 8 | checkpoint-8-2048 | 2.2545 | 0.9111 | 3.1656 | 2.4415 | 6.1165 | 0.3076 | 0.8670 |
| 9 | checkpoint-9-2304 | 2.1484 | 0.9086 | 3.0571 | 2.4600 | 6.1174 | 0.2923 | 0.8675 |
| 10 | checkpoint-10-2560 | 2.3216 | 0.9205 | 3.2421 | 2.7439 | 6.1170 | 0.3030 | 0.8669 |
| 11 | checkpoint-11-2816 | 2.2193 | 0.8512 | 3.0705 | 2.4324 | 6.1175 | 0.3224 | 0.8608 |
| 12 | checkpoint-12-3072 | 2.3883 | 0.9107 | 3.2989 | 2.3625 | 6.1193 | 0.3326 | 0.8651 |

## Observed Problems

1. Text generation did not improve over the ADAPT reproduction baseline.
   - V2 epoch 1/2/4 stayed around `CIDEr_des+exp=2.05-2.08`.
   - ADAPT reproduction reached `3.16` by epoch 4 and `3.30` by epoch 12.

2. Control metrics were severely misaligned or broken in V2 compared with ADAPT reproduction.
   - ADAPT reproduction course RMSE stayed around `6.12`.
   - Current V2 course RMSE stayed around `88.95`.
   - This is not a normal small regression; it indicates control scale, data bridge, resume state, or evaluation path mismatch.

3. The best-checkpoint update logic showed a likely resume-history issue.
   - The inspected run wrote best checkpoint files after epoch 4 even though epoch 2 text sum was higher than epoch 4.
   - Likely cause: best-selector state was seeded from `checkpoint_latest` rather than from full historical `metrics_summary.jsonl`.
   - This is recorded as an unresolved code issue unless separately patched.

4. Semantic/traffic-flow audit became partially informative but did not prove model control usage.
   - `pred_speed_delta_corr` and `pred_course_delta_corr` became non-null after audit fixes.
   - However correlation values were negative and the prediction delta standard deviations remained very small, so this does not prove a useful traffic-flow-to-control mechanism.

5. Speed and memory behavior changed after fast relaunch.
   - `batch_size=32`, `num_workers=8`, `gradient_accumulation_steps=1` filled GPU memory to about 40GB and raised GPU utilization.
   - This improved hardware utilization but did not solve metric quality.

## Interpretation

The current result is not evidence that ACPR FlowCal V2 improves ADAPT. It is evidence that the current V2 implementation/run path is still not aligned enough with the ADAPT reproduction state, especially for control evaluation and for preserving the text generation quality of the checkpoint it resumed from.

The most likely failure points are:

1. Resume/initialization did not preserve the strong text-generation state in a way that survived the V2 stage schedule.
2. Stage 1/2 frozen-prefix logic or trainable module selection may be too narrow for recovering text quality, while still allowing newly inserted modules to perturb hidden states.
3. Control head/eval bridge has a scale or target mismatch, because course RMSE jumping from about `6` to about `89` is too large to treat as ordinary training noise.
4. Best checkpoint selectors need to be seeded from previous metric history, not only the latest checkpoint payload.

## Decision

Do not continue this training run. Before another full run:

1. Run a pure evaluation of the intended resume checkpoint through the exact V2 eval bridge.
2. Prove V2 eval bridge reproduces ADAPT checkpoint metrics before enabling any V2 modules.
3. Fix best-checkpoint history seeding.
4. Fix or explain the control course scale mismatch.
5. Only then restart staged V2 training.

## ACPR-DynFlow V1 implementation findings (2026-06-23 03:45 China time)

### Verified facts
- Source worktree was cleaned before creating the DynFlow worktree. Seventeen untracked scratch/smoke/debug files were moved into `.background_runs/pre_dynflow_snapshot_20260623_030241/untracked`; they were not deleted.
- New worktree was created at `E:/sbw/FATE_Drive/fate_x_acpr_dynflow_v1_worktree` from source HEAD `8a52f4e99e81406cd949afaadb16c7483cf5025d`.
- Remote branch `acpr_dynflow_v1` was created and pushed from that base.
- Exact 32 predicate names were found in `E:/sbw/FATE_Drive/fate_oia_acpr_calalign_v1_2_worktree/configs/acpr_scene_predicates.yaml`.
- No complete ACPR-CalAlign predicate-query/prototype checkpoint was found. The small `ckp/classifier.pth.tar` found under the OIA worktree is not a valid substitute for the required formal query checkpoint.

### Verification results
- `python -m compileall -q fate_x tests/acpr_dynflow`: passed.
- `python -m pytest tests/acpr_dynflow -q`: `48 passed in 53.18s`.
- `run_acpr_dynflow_preflight --synthetic`: generated all required JSON reports and returned `passed=false`, `blockers=["dirty_worktree","oia_checkpoint_unresolved"]`, `missing_reports=[]`.
- Synthetic train smoke returned an actual batch line: `ACPR_DYNFLOW_BATCH ... "frames_shape": [1, 32, 3, 224, 224]`.
- The second smoke after the `torch.load(weights_only=False)` fix completed without the earlier FutureWarning.

### Important limitations; do not misreport these as complete
- The current implementation is a functional formal scaffold with real tensor pathways and tests, but it is not yet a fully verified paper-grade implementation of the whole DynFlow plan.
- The required BDD-OIA ACPR-CalAlign predicate-query checkpoint remains unresolved. This is a formal hard blocker.
- The current generated `video_backbone.py` is an independent lightweight video module used to validate the direct-image path; it does not yet instantiate and load an actual Kinetics Video Swin backbone.
- The current generated text decoder is lightweight and independent; it does not yet prove full BERT-base top-layer integration.
- The OIA predicate transfer path currently validates ontology names and checkpoint-path handling, but cannot validate true query/prototype transfer until the required checkpoint is supplied.
- Some tests are contract/presence tests. They prevent missing entrypoints, but they are not sufficient alone to prove scientific equivalence.

### Root cause of remaining formal block
- This is not a syntax/training-loop blocker. The remaining scientific blocker is missing required external artifact: the formal BDD-OIA ACPR-CalAlign predicate-query checkpoint.
- Because the plan forbids substituting unrelated checkpoints, formal training must remain blocked until that artifact is provided or the plan is explicitly revised.

## ACPR-DynFlow V1 real-loader findings (2026-06-23 04:16 China time)

### Assets resolved
- Video Swin K600 checkpoint downloaded from the official SwinTransformer release URL:
  `https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth`
- Local checkpoint size after download: `382,579,368` bytes.
- BERT-base downloaded from Hugging Face repo `bert-base-uncased` into `models/captioning/bert-base-uncased`.
- BERT directory now contains config/vocab and a model weight file.

### Root-cause fixes
- Video Swin was previously only path-recorded. It now uses actual `swin3d_b` forward.
- The official Video Swin checkpoint did not share exact key names with torchvision, but tensor shapes matched. A deterministic converter maps the official checkpoint to torchvision.
- BERT initially failed with protobuf descriptor errors. The fix sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` immediately before importing `transformers.BertModel`; this avoids downgrading global packages in `sbw39`.

### New verification evidence
- `run_acpr_dynflow_preflight --device cuda --synthetic` after loader hardening removed `bert_not_loaded`.
- `backbone_audit.json` reported:
  - `kinetics_loaded=true`
  - `uses_torchvision_swin=true`
  - `converted_keys=351`
  - `converted_ratio=0.9943342776`
  - only skipped keys: `cls_head.fc_cls.weight`, `cls_head.fc_cls.bias`
- Full tests after the real-loader change:
  - `python -m compileall -q fate_x tests/acpr_dynflow`
  - `python -m pytest tests/acpr_dynflow -q`
  - Result: `48 passed, 102 warnings in 295.85s`

### Remaining blocker
- `oia_checkpoint_unresolved` remains. This is now the only substantive formal training blocker after commit, assuming the worktree is clean.

## 2026-06-23 发现与根因：OIA checkpoint 和 gradient gate 不是表面问题

### OIA predicate transfer 根因
- 原状态：配置中 `oia_acpr_checkpoint` 是 unresolved placeholder；早期 preflight 只证明路径或字段存在，不能证明 OIA predicate queries 被真实加载。
- 搜索结果：在 `fate_oia_acpr_calalign_v1_2_worktree` 找到有效 checkpoint，包含 `model/predicate_head.predicate_queries` 和相关 query/key/value/logit/temperature 权重。
- 采用 checkpoint：`E:/sbw/FATE_Drive/fate_oia_acpr_calalign_v1_2_worktree/.background_runs/acpr_calalign_v1_2_resume_e15_17_sched28_20260616_125105/checkpoint_best_test_final_calibrated.pth`。
- 动态 preflight 证据：`oia_predicate_transfer_audit.json` 记录 `oia_loaded=true`、`oia_source=model`、`oia_source_dim=384`、`oia_prior_shape=[32,384]`、`checkpoint_sha256=84d3744a7505cca19b33ac2b517b58d71c98fd580f162dec4a6eee2aee1f64b2`。

### Gradient-chain gate 根因
- 原 gate 错误：`run_acpr_dynflow_preflight.py` 用 `all(v > 0 for every trainable parameter)` 判定，计划要求是每个 intended trainable component 有梯度，不是每个参数都非零。
- 更重要的是，排查发现有真实的 dead trainable path：
  - `lane_flow.encoder.*` 无梯度，因为 `MesoscopicLaneFlow.forward()` 产生的 `tokens` 没有被 `TrafficStateReasoner` 使用。
  - `reasoner.lateral.*` 无梯度，因为 `lateral_bias` 只被写进 dataclass，没有进入 decision/course prediction 或 loss。
  - `backbone.text_proj.*` 无梯度，因为 `bb.text_visual_tokens` 没有进入 text decoder。
  - `backbone.swin.head.*` 和 `text_decoder.bert.pooler.*` 是未使用头，却保持 trainable。
- 代码修复：
  - `traffic_state_reasoner.py` 将 `lane_flow["tokens"].mean(2)` 加入 `factor_tokens`。
  - `decision_ledger.py` 将 `flow.lateral_bias` 时间对齐后加入 course contribution。
  - `model.py` 将 `visual_tokens=bb.text_visual_tokens` 传给 `DynFlowTextDecoder`。
  - `text_decoder.py` 将 visual tokens 插入 caption hidden state；冻结 BERT pooler。
  - `video_backbone.py` 冻结 Swin classifier head。
  - `run_acpr_dynflow_preflight.py` 增加 component-level gradient report，包含 `component_norms`、`missing_components`、`missing_trainable_grad_params`、`frozen_params_with_grad`。
  - `audit_acpr_dynflow.py` 增加 `failed_required_reports` blocker。

### 当前验证结果
- 动态 preflight：`.background_runs/acpr_dynflow_v1_gradient_fixed_preflight/review_report.json` 只剩 `dirty_worktree`，`missing_reports=[]`，`failed_reports=[]`。
- Gradient chain：`gate_gradient_chain.passed=true`，`missing_components=[]`，`frozen_params_with_grad=[]`，`missing_trainable_grad_params=0`。
- 关键组件梯度均非零：`oia_query_mapper`、`predicate_query_residual`、`predicate_gru`、`predicate_visual_projection`、`mesoscopic_lane_flow`、`traffic_state_reasoner`、`response_lag`、`global_decision_stream`、`decision_ledger`、`visual_text_projection`、`text_decoder_top_layers`、`video_swin_trainable`。
- 测试：`compileall=0`；`pytest tests/acpr_dynflow -q` 为 `48 passed, 102 warnings in 313.56s`；`git diff --check=0`。

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

## Unified Findings Expansion - 2026-06-23 05:09:06 +08:00

### A. ADAPT reproduction is the local reference, not just the paper table
- The ADAPT paper numbers are useful context, but the most relevant baseline for ACPR-family modules is the user's local ADAPT reproduction on the same downloaded/preprocessed BDD-X data.
- Existing records indicate the local ADAPT reproduction reached approximately CIDEr_des+exp=3.16 by epoch 4 and around 3.30 by epoch 12, while the later ACPR FlowCal V2 continuation stayed near 2.05-2.08.
- Therefore the key comparison is not “does V2 look plausible,” but “does V2 improve over the exact checkpoint it resumed from.” That condition was not met.

### B. The main V2 failure was not just training length
- V2 text quality was already far below the known local reproduction baseline early in the run.
- Control-course metrics showed a scale-level issue: recorded course RMSE around 88.95 versus ADAPT reproduction around 6.12 is too large to explain as ordinary model undertraining.
- The likely failure class is resume/eval/data bridge mismatch, control target scaling mismatch, or hidden-state perturbation from inserted modules, not just insufficient epochs.
- Future staged training must first run a pure eval of the intended resume checkpoint through the new code path; if that pure eval does not reproduce baseline metrics, training should not start.

### C. Discrete action proxy evaluation was the wrong main control metric for BDD-X/ADAPT
- BDD-X/ADAPT control is continuous speed/course prediction. Forcing speed/course into maintain/stop/straight/turn proxy recalls was not aligned with the ADAPT task contract.
- Those proxy metrics were removed from the main interpretation. If retained later, they should be marked diagnostic-only and never used for best checkpoint selection.
- Correct control selection should use legal continuous metrics such as speed/course RMSE or threshold accuracies (A0.5, A1, etc.) when available.

### D. Traffic-flow audit became informative but was not enough
- The audit began to expose macro/mesoscopic traffic-flow factors and their relation to labels.
- A key earlier gap was that model-prediction correlations (pred_speed_delta_corr, pred_course_delta_corr) were null; this meant the audit could show label association but not that the model used flow factors for predictions.
- After audit patching, prediction correlations became non-null, but the values and low prediction-delta variance did not prove a useful causal flow-control mechanism.
- The next design should require intervention/ablation evidence, not only correlation.

### E. DynFlow V1 resolved several real implementation blockers
- The OIA predicate initializer now loads a real ACPR-CalAlign checkpoint containing model/predicate_head.predicate_queries rather than unresolved placeholders.
- The official/available OIA checkpoint has source dimension 384; DynFlow uses a mapper into the model dimension and records SHA/source metadata.
- Video Swin changed from a lightweight placeholder to actual 	orchvision.models.video.swin3d_b with official Kinetics-600 weights converted into torchvision keyspace.
- BERT changed from a lightweight text path to local ert-base-uncased, with bottom layers frozen and top layers trainable.
- The gradient audit initially exposed real dead trainable paths. These were fixed by wiring lane-flow tokens into the reasoner, lateral bias into course contribution, and visual text tokens into the text decoder.

### F. DynFlow V1 review-pass caveat
- A clean review pass existed at an earlier commit, but later inspection found the trainer/evaluator still had a text-evaluation honesty gap.
- Specifically, 	rain_acpr_dynflow.py used a fake 	ext_score = 0.0, and eval_acpr_dynflow.py did not generate ADAPT-style caption metrics from model outputs.
- This means any “best text” checkpoint selected before the fix was not scientifically valid.
- The latest patch replaces fake text-score logic with real text-metric availability checks and blocks/suppresses best-text updates when generated text metrics are not available.
- Because that patch changes code after the prior review pass, the pass must be regenerated after commit.

### G. Repeated failure pattern to avoid
- Starting a new complex stage before the previous evaluation bridge is proven stable caused multiple wasted runs.
- Adding traffic-flow modules before proving baseline checkpoint metric preservation made it hard to distinguish module failure from resume/eval mismatch.
- Using paper numbers without checking local reproduction history led to misleading interpretations.
- Future work must follow this order: pure checkpoint eval -> smoke train/eval -> short staged run -> only then full training.
### Post-commit DynFlow review finding - 2026-06-23 05:17:35 +08:00
- The 0515 final preflight proved the current code path at HEAD $head had no review blockers, loaded the real OIA predicate queries, and had a valid gradient chain.
- The OIA report filename is oia_predicate_transfer_audit.json; using oia_predicate_transfer_report.json was only an inspection-script filename mistake, not a preflight failure.
- Because documentation itself changes HEAD, final pass evidence must always be regenerated after the last commit. This is now an explicit process requirement for future work.
## 2026-06-23 DynFlow 本轮失败/慢速根因记录

### 1. 无效训练的直接根因：`masked_pos` 语义解析错误

旧 formal run 在 commit `50505c1` 上启动后，训练日志显示 `explanation_text=0.0`，但配置中 explanation text loss 权重并不为 0。这不是模型没有学习到 explanation，而是代码没有正确监督 explanation token。

根因定位：

- BDD-X/ADAPT dataloader 输出的 `masked_pos` 是二值 mask，shape 为 `[B, 30]`。
- 旧 `DynFlowTextDecoder._masked_position_loss` 把 `masked_pos` 当成显式 token position list 使用。
- 结果是大量样本只在 token position 0/1 上计算 loss，后半 explanation token 没有进入有效监督。
- 这会导致训练看似在跑，但 explanation 分支不可能按计划学习。

修复方式：

- 文件：`fate_x/acpr_dynflow/text_decoder.py`
- 新逻辑：检测 `masked_pos` 是否为二值 mask；如果是，则转换为真实 masked token positions；再与 packed `masked_ids` 对齐计算 masked LM loss。
- action/explanation 分段 loss 现在按真实 token position 切分，不再依赖错误的 0/1 位置。
- 新测试：`tests/acpr_dynflow/test_text_decoder_masked_positions.py`

验证结果：

- targeted pytest：`tests/acpr_dynflow/test_text_decoder_masked_positions.py` 通过。
- `compileall fate_x/acpr_dynflow` 通过。
- `pytest tests/acpr_dynflow -q`：`54 passed`。
- 修复后 smoke/formal logs 中 `explanation_text` 为非零。

### 2. 本轮 formal run 不是指标失败，而是计算预算失败

修复后的 formal run：

- task：`acpr_dynflow_v1_full_e66bce9_20260623_0922`
- run dir：`G:\sbw\FATE_Drive\active_runs\acpr_dynflow_v1_formal_e66bce9_maskfix_schtasks_20260623_0922`
- commit：`e66bce98285883315b949250dd73ec17df3f3214`
- stopped at：epoch 0 batch `254/1639`
- checkpoint/eval：无，因为未完成第一轮。

关键日志片段说明：

- batch 251：`loss=13.7593`, `action_text=4.1367`, `explanation_text=4.6997`, `flow_residual_speed=0.001997`, `flow_residual_course=0.003323`
- batch 252：`loss=22.0781`, `action_text=4.0920`, `explanation_text=4.6160`
- batch 253：`loss=13.3351`, `action_text=3.8401`, `explanation_text=5.2892`

这些 loss 是有限值，且 explanation supervision 已经存在；问题不是 NaN/崩溃，而是速度无法接受。

### 3. 为什么一轮会接近 20 小时

之前误解点：`optimizer_steps_per_epoch=235` 很容易被误读成每轮 235 个 batch。但实际训练日志显示 `Total training steps 1639`，这是 dataloader micro-batches。因为 gradient accumulation 为 7，所以约 7 个 micro-batches 才对应一个 optimizer step。

当前慢速来自组合因素：

- 32-frame 224 输入仍走 Video Swin 视觉主干，视频编码计算重。
- batch size 10 已经把单卡显存推到约 `45.6G/49.1G`，不是显存没用上。
- 单 GPU 无法复制 ADAPT 原文多卡吞吐。
- 每个 epoch 必须跑完 1639 个 micro-batches 才能保存 epoch checkpoint 和做 eval，因此不到一轮时没有可比较指标。
- 在线文本 decode/eval 和额外 DynFlow 中间量记录会进一步增加 wall-clock。

结论：这轮不是 DynFlow 机制已经被证明无效，而是当前 full online training 方案过慢，无法在合理时间内完成一轮并产生评估。下一步应优先解决吞吐和实验粒度，而不是继续等待同一配置。

