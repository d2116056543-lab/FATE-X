---
name: acpr-flowcal-pp-implementation-audit
description: Blocking adversarial audit for the complete direct-image ACPR-FlowCal++ BDD-X implementation on FATE-X flowtrace_pmt_v1.
---

# ACPR-FlowCal++ V1 Implementation Audit Skill

## 1. Purpose

This skill is the formal training authorization gate for ACPR-FlowCal++ V1.

It must reject:

- incomplete modules;
- configuration-only features;
- dead gradients;
- unused reason supervision;
- direct flow-to-action/control shortcuts;
- legacy FlowTrace PMT paths;
- fake intervention values;
- placeholder visualization;
- image/video feature caches;
- validation evaluation;
- test-driven parameter fitting;
- detached/background training.

Formal training is forbidden unless this skill writes:

```text
.background_runs/acpr_flowcal_pp_v1_preflight/
REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt
```

The pass file is valid only for the exact clean local Git HEAD that equals the pushed GitHub `flowtrace_pmt_v1` HEAD.

Any code/config change invalidates it.

---

## 2. Required context

Before auditing, read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

Required repository state:

```text
worktree = E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree
branch = flowtrace_pmt_v1
```

No new worktree is expected.

Use independent reviewer role and Superpowers verification/systematic-debugging skills.

---

## 3. Required inputs

```text
repo_root
config_path
output_dir
device
expected_branch=flowtrace_pmt_v1
```

---

## 4. Git and provenance gate

Reject unless:

1. current path is the expected worktree;
2. branch is `flowtrace_pmt_v1`;
3. `git status --porcelain` is empty;
4. local HEAD equals:
   ```text
   git ls-remote github refs/heads/flowtrace_pmt_v1
   ```
5. `.background_runs`, datasets, checkpoints, logits, and visuals are ignored;
6. implementation plan, config, skill, and implementation manifest exist;
7. hashes are recorded;
8. no reviewed code differs from committed tree.

Write:

```text
git_head
github_remote_head
branch
worktree
dirty_status
plan_sha256
skill_sha256
config_sha256
implementation_manifest_sha256
```

---

## 5. Formal import-graph gate

Trace the formal trainer import graph.

Require:

```text
fate_x.engine.train_acpr_flowcal_pp
fate_x.acpr_flow.model
```

Reject if formal path imports or instantiates:

```text
TokenPMTAdapter
LogSinkhornTransport as primary transport
FlowTraceLoss
FlowTraceRenderer
FlowTraceAtlasBuilder legacy implementation
legacy learn-mask path
```

Legacy files may remain but must be unreachable.

Require formal resolved config:

```text
legacy_flowtrace_v1_enabled=false
legacy_token_pmt_enabled=false
legacy_sinkhorn_transport_enabled=false
learn_mask_enabled=false
```

---

## 6. Named-output gate

Inspect formal model and trainer.

Reject any positional tail parsing such as:

```python
outputs[-2]
outputs[-3]
```

Require typed outputs with named fields:

```text
action_text_loss
explanation_text_loss
control_loss
baseline_masked_logits
enhanced_masked_logits
control_base_prediction
control_final_prediction
auxiliary_loss
loss_components
bundle
```

Run a real batch and verify every field type/shape.

---

## 7. Direct-image / no-cache gate

Formal training must show:

```text
frames shape [B,32,3,224,224]
direct_image_training=true
feature_cache_enabled=false
token_cache_enabled=false
build_cache_before_training=false
```

Reject if code:

- creates Video Swin/DINO feature shards;
- loads a learned visual feature cache;
- bypasses frame decoding;
- uses cached logits for formal training.

Online text target construction is allowed because it uses current input token embeddings and does not save a cache.

Record file-access evidence.

---

## 8. Backbone and multiscale gate

Verify:

1. one Video Swin forward per batch;
2. fine and coarse features exist;
3. fine resolution > coarse resolution;
4. dimensions are inferred, not hardcoded;
5. fused grid finite;
6. dense control tokens remain original ADAPT final tokens;
7. checkpoint baseline keys load without unexpected baseline mismatch.

Record:

```text
fine_shape
coarse_shape
fused_shape
dense_shape
video_swin_forward_count
baseline_missing_keys
baseline_unexpected_keys
```

---

## 9. Predicate ontology gate

Require exact 32 names and order.

Reject:

- fewer/more predicates;
- anonymous track slots replacing predicates;
- missing region prior;
- hard Boolean-only bottleneck.

For each predicate require:

```text
token
probability
attention map
trajectory confidence
relative motion
```

Record exact names.

---

## 10. Local partial transport gate

Use synthetic translated feature grids and unmatched regions.

Require:

1. local neighborhood, not full `N x N`;
2. 25 neighbors plus dustbin for radius 2;
3. row mass including dustbin equals 1 within `1e-5`;
4. no column normalization;
5. common shift is recovered;
6. unmatched source puts mass in dustbin;
7. gradients reach matching projections;
8. changing frame order changes transport;
9. memory complexity report confirms no dense global matrix.

Record:

```text
transport_shape
row_mass_error
translation_error
unmatched_dustbin_mass
transport_grad_norm
```

---

## 11. Temporal predicate field gate

Verify output shapes:

```text
attention [B,T,32,H,W]
tokens [B,T,32,D]
logits/probs [B,T,32]
confidence [B,T,32]
relative_motion [B,T-1,32,2]
descriptor [B,32,D]
```

Verify:

- entmax produces exact zeros in a non-degenerate test;
- each map sums to one;
- region priors affect expected predicates;
- transported prior is used after time 0;
- current-frame visual score is always used;
- low confidence weakens beta;
- temporal reverse changes trend sign;
- global translation is suppressed in relative motion;
- gradients reach predicate queries and transport projections.

---

## 12. Dynamic descriptor gate

Require actual use of:

```text
now
history
trend
volatility
motion
confidence
```

Create synthetic sequences:

```text
increasing
decreasing
stable
oscillating
```

Require distinct descriptor outputs.

Reject mean-only pooling.

---

## 13. Flow factor gate

Require exact factor groups.

Regime:

```text
clear_open_flow
stable_following
dense_following
queue_congestion
```

Phase:

```text
forming
stable
releasing
oscillating
```

Source:

```text
traffic_signal
lead_vehicle_group
merge_lane_constraint
turn_intersection
vulnerable_obstacle_conflict
```

Require:

```text
tokens [B,13,D]
logits/probs [B,13]
flow-to-predicate attention [B,13,32]
flow evidence maps [B,T,13,H,W]
```

Verify factor evidence maps equal weighted predicate maps.

Verify grammar support and contradiction matrices are loaded and used.

Reject hard mutually-exclusive flat flow classification as the only representation.

---

## 14. Online reason target gate

Require no saved sentence-embedding cache.

Given GT action/reason tokens:

1. pool detached local BERT word embeddings;
2. normalize action and reason embeddings;
3. subtract action-direction projection;
4. normalize residual.

Verify:

```text
finite target
shape [B,768]
action-direction cosine reduced after residualization
no gradient into target embeddings
```

Reject GT text input during inference.

---

## 15. Free-text PU gate

Use explicit synthetic phrases.

Verify:

```text
"traffic is clear"
  road_clear positive
  road_crowded contradictory
  unrelated predicates unknown

"cars ahead are stopped"
  queue/lead predicates positive
  clear flow contradictory
  unrelated predicates unknown
```

Require:

- positive full weight;
- contradiction full or reliability weight;
- unknown low weight;
- unknown is not hard negative;
- gradients reach predicate/flow logits;
- raw action/justification metadata is preserved correctly by collate.

---

## 16. Reason memory gate

Require:

```text
32 local reason tokens
13 flow reason tokens
1 null reason token
total 46
hidden dim 768
```

Verify each flow token contains:

```text
flow token contribution
predicate support contribution
semantic-name contribution
```

Require explicit memory metadata and evidence lineage.

Reject:

- one pooled global reason token;
- reason probabilities only;
- direct flow logits as action residual.

---

## 17. Temporal HardPair gate

Synthetic queue tests must verify:

- same action/control/video + different reason can be mined;
- reason-similar sample is not a negative;
- contradictory predicate/flow signature contributes;
- pair count cap;
- queue size cap;
- active/inactive reasons logged;
- pair weighted contribution obeys 8% budget;
- raw pair loss is not double-added.

Real 128-sample mechanism run must show nonzero active pair rate.

---

## 18. Temporal SECA integration gate

Locate actual `BertForImageCaptioning` source with `inspect`.

Require order:

```text
BERT hidden
-> split text and image hidden
-> SECA text hidden only
-> recombine
-> LM head
```

Verify:

1. image hidden max diff = 0 under SECA;
2. action/reason text hidden may change;
3. entmax attention over 46 memory tokens;
4. null token present;
5. separate action/explanation gates;
6. `W_o` Xavier/nonzero initialized;
7. gate raw initialized zero;
8. zero gate exact baseline logits;
9. zero gate action/explanation gate gradients nonzero;
10. after one optimizer step q/k/v/out gradients nonzero;
11. action gradient into reason memory approximately 0.25 of unscaled reference;
12. explanation gradient is unscaled;
13. beam size 3 expands memory correctly;
14. inference uses no GT justification;
15. flow cannot bypass reason memory.

Reject the old triple-product PMT.

---

## 19. Control mediation gate

Require sensor head exposes base prediction and hidden states.

Verify:

```text
base [B,32,2]
final [B,32,2]
control hidden [B,32,768]
reason attention [B,32,46]
```

Require:

- control adapter reads reason memory only;
- zero gate exact base output;
- zero gate control gate gradient nonzero;
- invalid `-1` labels masked;
- no direct flow-to-control delta;
- bounded residual uses train signal scale;
- separate speed/course deployment scales.

---

## 20. Prefix-to-future gate

Full mode only.

Verify:

```text
input = first 24 frames
target = final 8 control steps
future control never enters state input
```

Temporal shuffle/reverse must worsen the prefix prediction on synthetic/real smoke.

---

## 21. Sequence-CalAlign gate

Require deterministic 10% train-calib IDs.

Verify:

- main weights frozen;
- base logits/predictions detached;
- zero alpha candidate exists;
- action/explanation/speed/course independently scaled;
- temperature separately fitted by segment;
- no test sample enters fitting;
- no test metric updates alpha or temperature;
- alpha zero exactly restores ADAPT;
- fitted values saved in checkpoint/manifest.

Reject any calibration fitted from test labels.

---

## 22. Loss gate

Require separate logged real tensors:

```text
action_text
explanation_text
control
predicate_pu
flow_pu
reason_semantic
future_control
hardpair_raw
hardpair_budgeted
trajectory
action_preserve
control_preserve
memory_diversity
total
```

Verify every configured nonzero weight contributes to total and intended gradients.

Reject:

- fallback zero-target dynamic control loss;
- latent-state intervention norm loss;
- legacy sparse-mask loss;
- double-counted base text/control loss.

---

## 23. Optimizer/freeze/scheduler gate

Build a parameter-to-group manifest.

Require every trainable parameter appears exactly once.

Verify LRs:

```text
predicate 1e-4
flow 1e-4
reason 1e-4
HardPair 5e-5
SECA 5e-5
control adapter 2e-5
future head 5e-5
BERT last2 1e-5
Swin last stage 5e-6
sensor 1e-5
```

Verify stage freeze transitions actually change `requires_grad`.

Verify warmup + cosine acts on every group.

---

## 24. ADAPT fallback gate

Use the released checkpoint and identical input.

With deployment alphas zero:

```text
action teacher-forced logits
action generation
explanation teacher-forced logits
explanation generation
speed/course prediction
```

must match ADAPT within:

```text
CPU/unit: 1e-6
CUDA/BF16: 1e-4
```

The new modules must execute; equivalence may not be achieved by skipping construction.

---

## 25. Intervention gate

For every intervention, rerun all downstream computations.

### State off

Must change/recompute:

```text
reason memory
SECA
control adapter
outputs
```

### Predicate off

Must change/recompute:

```text
flow
reason memory
outputs
```

### Evidence tube off

Must change fused-grid evidence, then rerun:

```text
predicate
flow
reason
outputs
```

### Random equal mass

Verify exact patch-count or cumulative-mass equality.

### Temporal shuffle/reverse

Must rerun the state branch; not merely reorder a diagnostic tensor.

Teacher-forced action/reason deltas must be real.

---

## 26. Visualization gate

A real sample must produce:

```text
named predicate tube panel
flow regime/phase/source curve panel
hierarchical support graph
token intervention effect panel
control base/enhanced/state-off panel
counterfactual twin
JSON with source tensors and checkpoint SHA
```

Reject:

- one grayscale map;
- manual boxes;
- random/fabricated token effects;
- templated counterfactual text;
- Atlas that only dumps records.

---

## 27. Test-only protocol gate

Require:

```text
no validation DataLoader
test evaluation once after every epoch
best split = test
protocol tag = test_selected_user_requested
```

Verify test does not update:

```text
weights
optimizer
scheduler
HardPair queue
PU rules
Sequence-CalAlign
```

---

## 28. Formal experiment-suite gate

Resolved suite must contain:

```text
Run 0 ADAPT eval
Common Stage A 6 epochs
Fork B1 12 epochs + 3 calibration
Fork B2 12 epochs + 3 calibration
no-retrain intervention suite
Canvas
Atlas
```

No metric-based early stop.

---

## 29. Precision and nonfinite gate

Require runtime BF16 support check.

If BF16:

```text
autocast bfloat16
no GradScaler
```

If FP16 fallback:

```text
GradScaler
initial scale <= 4096
overflow counter
finite guard
```

Reject unexplained TensorBoard NaN/Inf warnings.

---

## 30. Memory gate

Run 20 full steps for candidate micro-batches.

Require:

```text
direct images
formal losses
no cache
no OOM
no NaN/Inf
peak reserved <= 43 GiB
largest stable candidate selected
```

Report preferred 30-42 GiB range honestly.

Do not allocate dummy memory.

---

## 31. Foreground supervisor gate

Static scan forbids:

```text
Start-Process
Start-Job
schtasks
nohup
shell &
DETACHED_PROCESS
hidden process flags
```

Dynamic smoke requires:

- attached child;
- stdout/stderr streaming;
- heartbeat;
- epoch artifact verification;
- no metric stop;
- OOM fallback;
- resume latest;
- explicit user sentinel;
- review-pass SHA verification;
- remote SHA verification.

---

## 32. Real smoke and mechanism gates

### 8-step direct-image smoke

Require:

```text
frames [B,32,3,224,224]
forward/backward
test eval
checkpoint
no cache
finite values
```

### Gradient chain

Require nonzero:

```text
predicate query
transport projection
flow query
reason memory
SECA gates
control gates
```

### 128-sample mechanism overfit

Require decreases/improvements in intended losses and no collapse.

### Temporal necessity

Require phase/trend and future-control sensitivity.

### Real interventions

Require evidence deletion effect greater than random on average.

---

## 33. Required report files

Write:

```text
review_report.json
implementation_manifest.json
formal_import_graph.json
tensor_contracts.json
gradient_report.json
fallback_equivalence.json
pu_target_audit.json
hardpair_audit.json
sequence_calalign_audit.json
intervention_audit.json
visual_artifact_index.json
memory_probe.json
supervisor_audit.json
```

---

## 34. Review pass file

Only on full pass write:

```text
REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt
```

It must contain:

```text
timestamp
reviewer
worktree
branch
git_head
github_remote_head
dirty_status
plan_hash
skill_hash
config_hash
formal_import_graph_hash
all test commands/results
direct-image proof
no-cache proof
named-output proof
predicate names
flow factor names
transport proof
temporal proof
PU proof
reason-memory proof
HardPair proof
SECA hook location
zero-gate equivalence
zero-gate gradient proof
post-step gradient proof
control mediation proof
Sequence-CalAlign no-test-fit proof
intervention proof
visual proof
test-only proof
memory selection
foreground proof
```

Authorization statement:

```text
ACPR_FLOWCAL_PP_V1_IMPLEMENTATION_REVIEW_PASS
The reviewed local HEAD equals the pushed GitHub flowtrace_pmt_v1 HEAD.
The worktree is clean.
All architecture, gradient, fallback, direct-image, no-cache, PU, HardPair,
SECA, control, calibration, intervention, visualization, test-only,
memory, and foreground-supervision gates passed.
The formal experiment suite is authorized for this exact commit.
```

If any gate fails:

- remove any old pass file;
- write exact blockers;
- return to Agent A;
- do not train.
