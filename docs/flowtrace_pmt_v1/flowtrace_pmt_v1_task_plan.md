# FlowTrace PMT V1 Task Plan

Generated: 2026-06-18 23:55 CST
Remote repo: `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree`
Local mirror target: `C:\Users\WLJTXY\Downloads\FATE_X_FlowTrace_PMT_V1_Package_20260618`

## Current Contract

This document is FlowTrace-specific and intentionally does not use the generic names `task_plan.md`, `findings.md`, or `progress.md` in the local Downloads folder.

The active contract remains the strict FlowTrace PMT V1 plan:

- Real-data integrated smoke must use direct image batches, not learned-feature cache.
- Smoke must run bounded train/eval samples and prove forward/backward behavior.
- Smoke must produce decode evidence, state-off and equal-mass random intervention evidence, FlowTrace Canvas, schema artifacts, checkpoint_latest, and finite/nonzero gradients for required modules.
- Formal training must not start until `REVIEW_PASS_FLOWTRACE_PMT_V1.txt` exists and binds the exact implementation state.

## Current State

Status: **formal training blocked**.

The six-hour smoke issue has been fixed, but strict gate is still failing.

### Completed

1. Diagnosed smoke runtime issue.
2. Added hard smoke train-step limiter:
   - `fate_x/engine/flowtrace_adapt_bridge.py` passes `--flowtrace_max_train_steps`.
   - `src/tasks/run_adapt.py` parses it and caps `max_global_step`, `max_iter`, and `save_steps`.
   - Train loop exits with `FlowTrace hard smoke train-step limit reached: 8/8`.
3. Added regression tests:
   - `test_flowtrace_bridge_adds_hard_train_step_limit_for_real_smoke`
   - `test_flowtrace_smoke_train_step_limit_is_explicit_and_hard`
4. Verified targeted FlowTrace suite:
   - `24 passed, 10 warnings`
5. Re-ran real-data bounded smoke:
   - run dir: `.background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit`
   - stopped at `global_step=8`
   - total training time: `0:04:19.789397`

## Blocking Gate Items

The latest smoke summary still fails these strict requirements:

| Requirement | Current value | Gate status |
|---|---:|---|
| `grad_norms.reason_state_head > 0` | `0.0` | FAIL |
| `grad_norms.pmt > 0` | `0.0` | FAIL |
| `state_off_intervention == true` | `False` | FAIL |
| `random_equal_mass_intervention == true` | `False` | FAIL |
| no long smoke runtime | fixed, `8/8` stop | PASS |
| direct image batch | `img_feats [1,32,3,224,224]` | PASS |
| eval completed | `True` | PASS |
| checkpoint_latest | `True` | PASS |
| FlowTrace canvas | `True` | PASS |
| artifact schema | `True` | PASS |
| `no_nan_inf` summary flag | `True` | PASS with warnings to investigate |

## Next Required Steps

1. Trace why `reason_state_head` receives zero gradient in the real ADAPT smoke path.
2. Trace why `token_pmt_adapter` receives zero gradient in the real ADAPT smoke path.
3. Trace why intervention evidence remains false and deltas remain zero.
4. Decide whether TensorBoard `NaN or Inf found in input tensor` and DeepSpeed FP16 overflow are benign dynamic-loss-scale effects or evidence of bad logged values.
5. Re-run the same bounded smoke after fixes.
6. Only after strict smoke passes, run the strict audit and write `REVIEW_PASS_FLOWTRACE_PMT_V1.txt`.
7. Only then start full training.

## Do Not Do Yet

- Do not start full training.
- Do not mark implementation complete.
- Do not write `REVIEW_PASS_FLOWTRACE_PMT_V1.txt`.
- Do not ignore zero PMT/reason gradients just because the smoke exits normally.
