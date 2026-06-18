# FlowTrace PMT V1 Findings

Generated: 2026-06-18 23:55 CST

## Finding 1 - Six-hour smoke was a training-loop cap bug

The real-data smoke should not take six hours. The root cause was not dataset size or GPU speed. It was a wiring bug:

- `--max_steps 8` was mapped only to ADAPT `--limited_samples`.
- `src/tasks/run_adapt.py` still computed `max_iter` and `max_global_step` from the dataloader length.
- Therefore the smoke entered a near-full ADAPT training schedule and produced a multi-hour ETA.

Evidence from fixed smoke:

```text
06/18/2026 23:34:20 - INFO - __main__ -   FlowTrace hard smoke train-step limit enabled: 8; capped max_global_step=8, max_iter=8
06/18/2026 23:38:43 - INFO - __main__ -   FlowTrace hard smoke train-step limit reached: 8/8
06/18/2026 23:38:43 - INFO - __main__ -   Total training time: 0:04:19.789397 (32.4737 s / iter)
```

## Finding 2 - Smoke now runs bounded, but strict gate still fails

The run completes, evaluates, writes checkpoints, emits canvas/schema artifacts, and stops at step 8. However, it is not a valid pass for the plan because required gradient and intervention evidence is missing.

Current smoke summary:

```json
{
  "real_data_smoke": true,
  "direct_image_training": true,
  "feature_cache_enabled": false,
  "token_cache_enabled": false,
  "forward_backward": true,
  "train_samples": 8,
  "batch_shapes": {
    "input_ids": [
      1,
      30
    ],
    "attention_mask": [
      1,
      814,
      814
    ],
    "token_type_ids": [
      1,
      30
    ],
    "img_feats": [
      1,
      32,
      3,
      224,
      224
    ],
    "masked_pos": [
      1,
      30
    ],
    "masked_ids": [
      1,
      45
    ],
    "car_info": [
      1,
      2,
      32
    ]
  },
  "grad_norms": {
    "transport": 9.137259483337402,
    "track_queries": 3.6494784355163574,
    "state_composer": 0.3745255167305004,
    "reason_state_head": 0.0,
    "pmt": 0.0
  },
  "no_nan_inf": true,
  "decoder_logprobs": true,
  "decoder_token_logprobs_head": [
    -10.068157196044922,
    -9.894997596740723,
    -10.626166343688965,
    -10.97802448272705,
    -10.23315715789795,
    -10.043126106262207,
    -10.921133995056152,
    -11.562376976013184,
    -10.465731620788574,
    -9.496623992919922,
    -10.623305320739746,
    -9.800065994262695
  ],
  "state_off_intervention": false,
  "random_equal_mass_intervention": false,
  "intervention_state_off_delta": 0.0,
  "intervention_equal_mass_delta": 0.0,
  "artifact_schema": true,
  "artifact_schema_dir": ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/epoch_000",
  "flowtrace_canvas": true,
  "flowtrace_canvas_path": ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/flowtrace_smoke_visuals/training_06d501fd-fd237e38_00007_flowtrace_canvas.png",
  "eval_completed": true,
  "eval_files": [
    ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-0-1/pred.BDDX_des.testing_32frames.beam1.max15.eval.json",
    ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-0-1/pred.BDDX_exp.testing_32frames.beam1.max15.eval.json",
    ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-1-8/pred.BDDX_des.testing_32frames.beam1.max15.eval.json",
    ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-1-8/pred.BDDX_exp.testing_32frames.beam1.max15.eval.json"
  ],
  "eval_samples": 8,
  "checkpoint_latest": true
}
```

Critical failure fields:

```text
reason_state_head grad norm = 0.0
pmt grad norm = 0.0
state_off_intervention = False
random_equal_mass_intervention = False
intervention_state_off_delta = 0.0
intervention_equal_mass_delta = 0.0
```

## Finding 3 - Direct-image path is active

Evidence:

```text
input_ids = torch.Size([1, 30])
img_feats = torch.Size([1, 32, 3, 224, 224])
car_info = torch.Size([1, 2, 32])
```

This confirms the smoke is not using a learned feature cache. It is using direct video/image tensor batches.

## Finding 4 - There are numeric warnings that need investigation before full training

The smoke summary reports `no_nan_inf: true`, but the log contains repeated TensorBoard warnings and one DeepSpeed dynamic loss-scale overflow:

```text
06/18/2026 23:34:40 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
[2026-06-18 23:36:56,643] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 4
[2026-06-18 23:36:56,643] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 65536.0, reducing to 32768.0
```

Interpretation:

- One FP16 overflow can be normal under DeepSpeed dynamic loss scaling.
- Repeated TensorBoard NaN/Inf warnings are not acceptable to ignore without tracing which scalar is non-finite.
- Formal training should wait until the logged non-finite source is identified or explicitly proven benign.

## Finding 5 - Signal evaluation was unavailable for the 8-sample eval subset

The smoke wrote:

```text
signal evaluation skipped: no valid control rows; wrote ...signal_unavailable.json
```

This means the first eight test rows did not contain valid control rows for signal evaluation. This is not the same as action/caption eval failing, but it is important evidence for why signal metrics are absent in smoke.

## Finding 6 - Current repository is still dirty and review pass is missing

Remote status:

```text
---GIT STATUS---

 M configs/flowtrace_pmt_v1_bddx_32f_224.yaml

 M fate_x/engine/adapt_live_decoder_wrapper.py

 M fate_x/engine/audit_flowtrace_pmt_implementation.py

 M fate_x/engine/build_reason_state_anchors.py

 M fate_x/engine/eval_flowtrace_pmt.py

 M fate_x/engine/probe_flowtrace_memory.py

 M fate_x/engine/supervise_flowtrace_foreground.py

 M fate_x/engine/train_flowtrace_pmt.py

 M fate_x/explain/flowtrace_renderer.py

 M fate_x/losses/flowtrace_losses.py

 M fate_x/models/sinkhorn_transport.py

 M fate_x/models/token_pmt_adapter.py

 M src/configs/config.py

 M src/datasets/vl_dataloader.py

 M src/modeling/load_swin.py

 M src/modeling/multitask_e2e_vid_swin_bert.py

 M src/tasks/run_adapt.py

 M src/utils/deepspeed.py

 M tests/test_flowtrace_config_contract.py

 M tests/test_flowtrace_e2e_smoke.py

 M tests/test_flowtrace_live_decoder.py

 M tests/test_flowtrace_sinkhorn_transport.py

 M tests/test_flowtrace_token_pmt.py

?? docs/superpowers/

?? fate_x/engine/backbone_output_utils.py

?? fate_x/engine/checkpoint_utils.py

?? fate_x/engine/flowtrace_adapt_bridge.py

?? tests/test_flowtrace_dynamic_traffic_state_composer.py

?? tests/test_flowtrace_losses.py

?? tests/test_flowtrace_real_smoke_evidence.py

?? tests/test_flowtrace_strict_audit_contract.py

---STRICT FILES---



Name                                              Length LastWriteTime

----                                              ------ -------------

checkpoint-0-1                                           2026/6/18 23:35:55

checkpoint-1-8                                           2026/6/18 23:38:13

checkpoint_best                                          2026/6/18 23:36:28

checkpoint_latest                                        2026/6/18 23:38:18

epoch_000                                                2026/6/18 23:34:40

flowtrace_smoke_visuals                                  2026/6/18 23:37:00

log                                                      2026/6/18 23:33:53

tokenizer                                                2026/6/18 23:34:23

datasets_BDDX_testing_32frames.yamleval_logs.json 614    2026/6/18 23:38:13

flowtrace_adapt_command.json                      3032   2026/6/18 23:33:15

flowtrace_real_smoke_summary.json                 2308   2026/6/18 23:38:43

run_manifest.json                                 2393   2026/6/18 23:33:15

smoke.log                                         45424  2026/6/18 23:38:43

train.log                                         45424  2026/6/18 23:38:43





---REVIEW PASS---

MISSING


```

`REVIEW_PASS_FLOWTRACE_PMT_V1.txt` is missing, so full training remains blocked by process contract.

## Hypotheses for Gate Failure

1. `reason_state_head` may not be connected to a loss term in the actual ADAPT training forward path.
2. `token_pmt_adapter` may be used for inference/artifact emission but detached or not used by the training loss.
3. Intervention deltas may be computed on a bundle that does not contain active state/probability perturbation fields, causing both intervention flags to remain false.
4. Intervention loss currently may be logged but not weighted into the training objective; if so, it cannot create gradients by itself.
5. TensorBoard non-finite warnings may come from grad norm / skipped FP16 step / scalar conversion rather than primary loss, but this must be traced.

## Required Debug Targets

- Inspect FlowTrace bundle fields written by `MultitaskVideoTransformer.forward`.
- Verify `reason_state_head` output participates in `flowtrace_losses.py` with nonzero weight or has a dedicated smoke gradient probe.
- Verify PMT output participates in the training loss path, not only artifact rendering.
- Make intervention evidence real: state-off and equal-mass random intervention must alter logits/probabilities enough to set boolean evidence true.
- Add a scalar finite-check around TensorBoard logging to identify the exact non-finite source.
