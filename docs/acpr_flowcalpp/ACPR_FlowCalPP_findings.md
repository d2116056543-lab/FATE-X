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
