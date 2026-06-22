# ACPR FlowCalPP / FlowCal V2 Unified Findings

Updated: 2026-06-23 00:25:13 Asia/Shanghai

## 1. Final Answer For Current Question

The current V2 run did not beat the ADAPT reproduction epoch 4. It also did not beat the ADAPT reproduction best checkpoint.

| Run | CIDEr_des | CIDEr_exp | CIDEr_sum | CIDEr_sum x100 | speed RMSE | course RMSE |
|---|---:|---:|---:|---:|---:|---:|
| ADAPT repro epoch4 | 2.3225 | 0.8410 | 3.1634 | 316.3 | 2.9790 | 6.1208 |
| ADAPT repro best epoch12 | 2.3883 | 0.9107 | 3.2989 | 329.9 | 2.3625 | 6.1193 |
| FlowCalPP V1 strict epoch14 | 0.6488 | 0.5441 | 1.1930 | 119.3 | 2.5398 | 6.1176 |
| FlowCalPP V1 historical text best | 1.4878 | 0.7872 | 2.2750 | 227.5 | 2.6372 | 6.1163 |
| FlowCal V2 best epoch2 | 1.3208 | 0.7553 | 2.0761 | 207.6 | 7.0209 | 88.9550 |
| FlowCal V2 latest epoch4 | 1.3210 | 0.7474 | 2.0685 | 206.8 | 6.9460 | 88.9634 |

Direct comparison:

- V2 best vs ADAPT repro epoch4: `2.0761 - 3.1634 = -1.0873` raw CIDEr, about `-108.7` on paper x100 scale.
- V2 best vs ADAPT repro best: `2.0761 - 3.2989 = -1.2228` raw CIDEr, about `-122.3` on paper x100 scale.
- V2 control is not just slightly worse; course RMSE around `88.95` indicates a scale/path mismatch versus ADAPT reproduction around `6.12`.

## 2. What V1 Proved

V1 / FlowCalPP established the engineering infrastructure:

- Real epoch-end test evaluation was added.
- Latest checkpoint was saved before evaluation.
- Best checkpoint variants were added.
- ADAPT text metrics and continuous control metrics were separated.
- Discrete decision proxy was removed from the main metric path.
- Traffic-flow audit fields were added.
- Control metrics could be near ADAPT scale under the V1 path.

V1 did not prove text improvement over ADAPT:

- strict epoch14 text sum `1.1930` was far below ADAPT reproduction.
- historical text best `2.2750` was better but still below ADAPT reproduction epoch4 `3.1634`.

## 3. What V2 Proved

V2 did not prove improvement. It exposed deeper alignment issues:

1. Text quality dropped relative to ADAPT reproduction and did not recover in semantic recovery or early axis-aware motion.
2. Course control metric was on the wrong scale or wrong path.
3. Traffic-flow prediction correlations became non-null but remained weak/negative.
4. GPU fill and runtime were improved, but performance quality did not improve.
5. The intended staged plan was not enough to preserve the strong ADAPT reproduction state.

## 4. Important Engineering Problems Encountered

| Problem | Evidence | Fix/Action | Current Status |
|---|---|---|---|
| Smoke training did not guarantee full functionality | Earlier work could run but lacked proper eval/checkpoint behavior | Added per-epoch eval and checkpoint logic | Infrastructure improved |
| Discrete control proxy was invalid as main metric | Proxy collapsed to majority classes and was not ADAPT metric | Removed from main reporting | Resolved for reporting |
| Evaluation after training could lose progress | Eval could fail after an epoch | Save latest before eval | Resolved in trainer design |
| Traffic audit initially could only compare factors with labels | `pred_speed_delta_corr` / `pred_course_delta_corr` were null | Added prediction-delta correlation outputs | Partially resolved; values weak |
| GPU utilization was too low | low memory use around 7-12GB | fast relaunch: batch 32, workers 8, grad accum 1 | Runtime improved |
| V2 best checkpoint update appeared inconsistent | epoch4 wrote best files despite epoch2 better text sum | Root cause identified: selector likely seeded from latest payload, not full history | Needs fix before next run |
| V2 control course metric broke scale | V2 course RMSE around 89 vs ADAPT around 6 | Recorded as blocker | Unresolved |

## 5. Root Cause Ranking

Most likely causes of V2 failure:

1. No-op bridge parity was not proven before enabling V2 modules.
2. V2 resume path did not preserve or evaluate the ADAPT checkpoint in an ADAPT-equivalent way.
3. Control evaluation or target conversion has a scale mismatch, especially for course.
4. Stage 1/2 trainable prefix choices may perturb caption hidden states without enough recovery signal.
5. Best-checkpoint state needs full metric-history seeding.

## 6. Do Not Repeat

- Do not launch a long V2 run before verifying no-op ADAPT parity.
- Do not compare a modified eval bridge against ADAPT paper without first comparing against local ADAPT reproduction checkpoint outputs.
- Do not treat high GPU utilization as progress if metrics are off-scale.
- Do not let separate V1/V2 markdown files split the record again.
- Do not use `.tmp` checkpoints.
- Do not use discrete proxy as the main control metric.

## 7. Required Next Validation If Work Resumes

1. Load ADAPT reproduction epoch4 or epoch12 checkpoint through V2 bridge with all new modules disabled.
2. Expected: text and control metrics must match local ADAPT reproduction within a small tolerance.
3. If text matches but control does not, isolate `speed/course` target scaling and course angular/circular conversion.
4. If neither matches, fix the data/eval bridge before training.
5. Only after parity should staged FlowCal V2 modules be enabled.
