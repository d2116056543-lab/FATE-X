# ACPR FlowCalPP Findings

更新时间：2026-06-21 13:20 Asia/Shanghai

## 1. 主结论

- 当前训练已经停止在 epoch 15 中途；该停止点没有产生 test eval，不能作为结果对比。
- 最完整的正式评估点是 strict epoch14：`E:\sbw\FATE_Drive\active_runs\acpr_linux_b4w4_resume_envfix_20260621_065540\train\eval_epoch_014.json`。
- 历史文本最高点是 `historical_text_best` 的 `eval_epoch_005.json`，CIDEr 合计更高，但它属于旧诊断口径，不能代表当前 traffic-flow/circular-control 完整机制。
- Control 指标已经接近 ADAPT 表格水平；文本指标仍显著低于 ADAPT 原文，尤其 description CIDEr 差距大。
- 交通流机制已经有可审计输出：不仅有 traffic factor 与真实 control delta 的相关性，也有与模型 predicted control delta 的相关性；但 epoch14 是 course circular fix 前的结果，仍需新一轮 eval 验证修复后变化。

## 2. ADAPT 原文对比口径

来源：ADAPT 官方仓库 `https://github.com/jxbbb/ADAPT` 与论文 `https://arxiv.org/abs/2302.00673`。

ADAPT 论文表格常用数值按 x100 展示；这里换成代码 eval JSON 的小数口径：

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

解释：control 已经不是主要短板，text generation 是当前最大差距。strict epoch14 的 text CIDEr_des+exp 只有 ADAPT 约 34%；历史文本 best 约 65%。

## 3. strict epoch14 文本指标

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

## 4. strict epoch14 continuous control 指标

| Signal | RMSE | MAE | Acc@0.1 | Acc@0.5 | Acc@1 | Acc@5 | Acc@10 | valid_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| speed | 2.5398 | 1.7603 | 0.0691 | 0.2894 | 0.4492 | 0.9406 | 0.9958 | 67729 |
| course | 6.1176 | 0.8891 | 0.5514 | 0.8676 | 0.9102 | 0.9734 | 0.9878 | 67936 |

## 5. historical text best 文本指标

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

## 6. historical text best continuous control 指标

| Signal | RMSE | MAE | Acc@0.1 | Acc@0.5 | Acc@1 | Acc@5 | Acc@10 | valid_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| speed | 2.6372 | 1.8662 | 0.0543 | 0.2482 | 0.4247 | 0.9320 | 0.9955 | 67729 |
| course | 6.1163 | 0.8804 | 0.5895 | 0.8681 | 0.9107 | 0.9733 | 0.9878 | 67936 |

## 7. 交通流审计输出

strict epoch14 的 traffic audit 已经记录 `target_*_delta_corr` 和 `pred_*_delta_corr`，说明当前实现能同时看交通流因子与真实控制变化、模型预测控制变化的关系。

| Factor | mean | std | target_speed_corr | pred_speed_corr | target_course_corr | pred_course_corr |
|---|---:|---:|---:|---:|---:|---:|
| queue_congestion | 0.3902 | 0.3303 | -0.1941 | -0.6591 | -0.0148 | -0.5277 |
| clear_open_flow | 0.2001 | 0.2127 | 0.2832 | 0.4172 | 0.0185 | 0.5259 |
| traffic_signal | 0.2523 | 0.2375 | -0.1414 | -0.5146 | -0.0060 | -0.2787 |
| turn_intersection | 0.3110 | 0.1827 | 0.1689 | -0.2454 | 0.0046 | 0.2657 |
| lead_vehicle_group | 0.1745 | 0.1857 | 0.0042 | -0.0763 | 0.0114 | -0.2505 |
| dense_following | 0.0196 | 0.0218 | 0.0285 | 0.6360 | 0.0089 | 0.2369 |
| forming | 0.0219 | 0.0302 | 0.0199 | 0.5855 | 0.0030 | 0.1935 |

解读：

- `clear_open_flow` 与 target speed delta 正相关，也与 predicted speed/course delta 正相关，说明模型输出对开阔交通流有响应。
- `queue_congestion`、`traffic_signal` 与 target speed delta 负相关，也与 predicted speed delta 负相关，说明模型在减速/拥堵语义上已有响应。
- `lead_vehicle_group` target 相关性弱，但 predicted 相关性为负，说明模型可能过度依赖前车/车群提示，这需要后续 zero-out 或 counterfactual audit 验证。
- epoch14 的 course 相关性仍是 circular fix 前结果；`coursecircularfix3` 需要完成一个 eval 后才能判断 course wrap-around 修复是否有效。

## 8. 组件作用判断

| Component | 当前证据 | 判断 |
|---|---|---|
| ADAPT-aligned text eval | 产生 `des_metrics`/`exp_metrics` 与 `CIDEr_des_plus_exp` | 已落地，是主文本对比口径 |
| Continuous control eval | 产生 speed/course RMSE 与 threshold accuracy | 已落地，不能再用离散 proxy 代替主评估 |
| checkpoint best split | text/control/adapt_joint/test best 文件均在 strict run 中出现 | 已落地，满足续训和选模需求 |
| Traffic-flow audit | strict epoch14 输出 target 与 pred control delta corr | 已落地，但需要 course circular fix 后复测 |
| Flow/Predicate PU | batch loss 中持续有 `flow_pu`、`predicate_pu` | 已参与训练，但文本收益仍不足 |
| Reason semantic | batch loss 中持续有 `reason_semantic` | 已参与训练，但未证明能提升 CIDEr |
| Hardpair | batch 中 `hardpair_active_pair_rate=1.0` | 已启用，需要继续观察是否影响 text/control tradeoff |
| Future control | batch 中 `future_control` 有输出 | 已启用，但 tail 存在 spike，需要继续监控 |

## 9. 训练中出现的问题与修复

| 问题 | 现象 | 修改/处理 | 当前状态 |
|---|---|---|---|
| 只有文本 CIDEr，没有 control 指标 | 早期 eval 输出只有 CIDEr | 加入 ADAPT-aligned continuous control eval | 已修复 |
| 离散 decision proxy 坍缩 | speed 只 predict maintain，course 只 predict straight，macro recall 0.3333 | 不再作为主评估，改用 BDD-X 连续 control 指标 | 已降级为历史诊断 |
| eval 出错可能导致该轮训练白跑 | 训练后才评估 | 改为 eval 前先保存 latest | 已修复 |
| best checkpoint 单一 | 不方便区分 text/control/joint | 增加 best_text/best_control/best_adapt_joint/best_test | 已修复 |
| traffic audit 无法证明模型是否使用因子 | `pred_speed_delta_corr`/`pred_course_delta_corr` 曾为 null | 补充 predicted control delta 统计和相关性 | 已修复，epoch14 有数值 |
| course 角度 wrap-around | course delta 线性差分可能产生假大误差 | 加入 circular course delta 修复 | 代码已改，当前 run 未完成新 eval |
| hardpair projection 无梯度 | `hardpair_raw_loss` 有记录，但 `model.hardpair.proj.weight.grad` 为 `None` | 定位为无 BERT embedding fallback 使用 predicted state，导致测试降级路径没有 eligible pair；改为 fallback 使用 `reason_for_pair` target | 已修复，相关单测和完整 ACPR 测试通过 |
| 训练速度慢 | Windows/SSH/小 batch/评估耗时大 | 切到 WSL/Linux、batch 4、workers 4，并保留后台任务 | 已改善，但每轮仍受 eval 和 caption decoding 影响 |
| 中途停止留下 tmp | 停止时正在写 ckpt | 标记 `.tmp` 为无效，仅用 `checkpoint_latest.pth` | 已记录 |

## 10. 当前不足

- strict epoch14 文本距离 ADAPT 仍大：description 更差，explanation 相对好一些。
- historical text best 比 strict epoch14 好，说明后续机制/权重可能牺牲了文本生成，需要重新平衡 text/control/flow loss。
- traffic-flow audit 目前证明“模型预测与因子相关”，但还没有完整的 intervention/zero-out 证明因果使用。
- current stopped run 没有完成 eval，不能判断 course circular fix 和后续训练是否提升。
