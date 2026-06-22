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
