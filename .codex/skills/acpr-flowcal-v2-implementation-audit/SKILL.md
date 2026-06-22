---
name: acpr-flowcal-v2-implementation-audit
description: Blocking adversarial authorization audit for the complete direct-image ACPR-FlowCal++ V2 BDD-X implementation on FATE-X flowtrace_pmt_v1.
---

# ACPR-FlowCal++ V2 Implementation Audit Skill

## 1. Authority and behavior

This skill is the only formal training authorization gate for ACPR-FlowCal++ V2.

It is adversarial. It must attempt to disprove implementation completeness. It must reject:

- files/classes that exist but are unreachable;
- YAML fields that never affect runtime;
- losses logged but not added;
- parameters included in optimizer but without intended gradients;
- gradients reaching a component from the wrong task;
- control paths that bypass the released ADAPT motion transformer;
- flow-to-control or flow-to-text shortcuts that bypass semantic reason memory;
- transport tensors that do not propagate predicate evidence;
- traffic-flow outputs that are only correlations or post-hoc labels;
- target leakage from `car_info`;
- static token embeddings presented as contextual reason targets;
- fixed negative treatment of unknown free-text predicates;
- equal-mean reason pooling;
- explanation improvements obtained by silently degrading control beyond contract;
- fake intervention deltas;
- placeholder visualization;
- image/video feature caches;
- validation evaluation;
- test-label calibration;
- detached/background execution;
- review passes for a different commit.

Formal training is forbidden unless this skill writes:

```text
.background_runs/acpr_flowcal_v2_preflight/
REVIEW_PASS_ACPR_FLOWCAL_V2.txt
```

The pass is valid only for the exact clean local Git HEAD equal to `github/flowtrace_pmt_v1`.

Any source, test, config, plan, launcher, or audit change invalidates it.

---

## 2. Required context and Superpowers

Before auditing, read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md

docs/acpr_flowcalpp/ACPR_FlowCalPP_task_plan.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_findings.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_progress.md

docs/runbooks/ACPR_FlowCal_V2_Implementation_Plan.md
configs/acpr_flowcal_v2_bddx_32f_224.yaml
.codex/skills/acpr-flowcal-v2-implementation-audit/SKILL.md
```

Use an independent reviewer role. Use installed Superpowers skills for:

```text
systematic-debugging
verification-before-completion
requesting-code-review or equivalent
```

Do not trust an implementer checklist without dynamic evidence.

---

## 3. Required repository state

Expected:

```text
worktree = E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree
branch = flowtrace_pmt_v1
remote = github
```

Reject unless:

1. current worktree is exact;
2. current branch is exact;
3. `git status --porcelain` is empty;
4. `git diff --check` passes;
5. local HEAD equals `git ls-remote github refs/heads/flowtrace_pmt_v1`;
6. `.background_runs`, checkpoints, datasets, generated logits, visual outputs, and caches are ignored;
7. plan/config/skill/implementation manifest exist;
8. all hashes are recorded.

Write `git_provenance.json`.

No new worktree or branch may be created by this audit.

---

## 4. Formal import-graph gate

Formal entrypoints must be:

```text
fate_x.engine.train_acpr_flowcal_v2
fate_x.acpr_flow_v2.model
```

Trace imports with AST and runtime `inspect`.

Reject if reachable formal code imports or instantiates:

```text
fate_x.acpr_flow.model.ACPRFlowModel
TokenPMTAdapter
LogSinkhornTransport
FlowTraceLoss
FlowTraceRenderer
legacy FlowTrace atlas
TinyDirectImageVideoBackbone
```

Legacy files may remain for historical evaluation but must be unreachable from the V2 trainer/evaluator/supervisor.

Write `formal_import_graph.json`.

---

## 5. Config binding gate

For every non-comment V2 YAML field, establish one of:

```text
runtime consumer path
audit-only field with explicit consumer
documentation-only field explicitly allowlisted
```

Reject orphan fields.

Dynamically mutate representative config values and prove runtime changes:

```text
transport radius
predicate beta max
unknown schedule
reason gradient scales
control residual bound
stage trainables
HardPair start epoch
scheduler type
SCST batch fraction
best selector tolerance
```

Write `config_binding_report.json`.

---

## 6. Direct-image and no-cache gate

Formal batch must be:

```text
frames [B,32,3,224,224]
```

Require:

```text
direct_image_training=true
feature_cache_enabled=false
token_cache_enabled=false
build_cache_before_training=false
```

Use file-open tracing during a real smoke. Reject if formal training loads:

- visual feature shards;
- precomputed Video-Swin/DINO tokens;
- cached logits;
- sentence embedding caches.

Streaming train control statistics and saving a small covariance/basis matrix are allowed. They are labels/statistics, not visual caches.

Write `file_access_audit.json`.

---

## 7. ADAPT text-contract gate

Resolve and record the source of:

```text
mask_prob
max_masked_tokens
max_seq_a_length
max_seq_length
use_sep_cap
attention-mask mode
```

Formal fallback values must include:

```text
mask_prob=0.5
max_masked_tokens=45
max_seq_a_length=15
max_seq_length=30
use_sep_cap=true
```

Compare V2 and original ADAPT dataloader tensors for identical train/test samples:

```text
input IDs
token type IDs
masked positions
masked IDs
attention-mask shape
raw action/justification metadata
```

Reject silent `0.15/20` formal settings.

Write `adapt_text_contract.json`.

---

## 8. Video baseline gate

On real direct frames verify:

- exactly one Video-Swin forward;
- released checkpoint loaded;
- fine/coarse native grids;
- temporal alignment to 32 for reasoning;
- dense original tokens preserved for ADAPT text/control;
- ADAPT fc loaded exactly;
- all tensors finite;
- fine spatial resolution greater than coarse;
- temporal native/aligned sizes recorded.

Write `video_backbone_report.json`.

---

## 9. ADAPT motion-baseline gate

Require formal use of released `Sensor_Pred_Head` weights.

Verify API:

```text
encode(img_feats)
predict(img_feats, frame_num)
```

Require:

```text
base prediction [B,32,2]
control hidden [B,32,768]
```

Target-leak test:

1. same video;
2. same explicit frame count;
3. two radically different `car_info` target tensors;
4. predictions and hidden must match within `1e-7` FP32 / `1e-4` BF16.

Zero-gate final control must match released ADAPT prediction.

Reject local reason-state linear baselines or synthetic temporal codes.

Write `adapt_motion_equivalence.json`.

---

## 10. Local partial transport gate

Synthetic and real tests must establish:

- 5×5 local candidates plus dustbin;
- common shift centers candidate neighborhoods;
- no wrapping borders;
- invalid candidates masked;
- row mass equals one;
- no column normalization;
- no dense `N×N` matrix;
- known translation recovered;
- unmatched region high dustbin mass;
- gradient reaches projection;
- `warp_source_map_to_current` moves a known map correctly.

Write `transport_audit.json`.

---

## 11. Transported predicate gate

Require exact 32 names/order.

For t>0 dynamically verify predicate attention depends on:

```text
current-frame visual score
region prior
warped previous attention
confidence-gated beta
```

Ablate each input separately and prove intended effect.

Reject if the `transport` argument is ignored.

Tensor shapes:

```text
attention [B,32,32,H,W]
tokens [B,32,32,D]
logits/probs/confidence [B,32,32]
relative motion [B,31,32,2]
descriptor [B,32,D]
```

Require entmax exact zeros and map normalization.

Write `predicate_trajectory_audit.json`.

---

## 12. Dynamic descriptor gate

Use increasing, decreasing, stable, and oscillating synthetic trajectories.

Require distinct and correct:

```text
now
history
trend sign
first difference
volatility
presence rate
motion mean/variance
confidence
```

Reject mean-only pooling.

Write `dynamic_descriptor_audit.json`.

---

## 13. Lane-flow-field gate

Require left/center/right soft masks affected by both geometry and drivable predicates.

Verify outputs:

```text
soft masks [B,32,3,H,W]
occupancy [B,32,3]
relative motion [B,32,3,2]
coherence/stopped/queue [B,32,3]
temporal tokens [B,32,3,D]
descriptor [B,3,D]
```

Synthetic cases:

```text
center queue forming
queue releasing
left lane freer
right lane blocked
temporal shuffle
temporal reverse
```

Require correct signs/order sensitivity.

Reject claims or variable names asserting physical traffic density/metric velocity.

Write `lane_flow_audit.json`.

---

## 14. Axis-aware flow gate

Require exact original 13 semantic factors plus:

```text
axis: longitudinal/lateral
direction: left/neutral/right
```

Verify signed left/right inputs affect direction.

Weak target test:

- derived from train controls only;
- targets detached;
- no control values enter forward state construction;
- circular course delta used;
- changing labels does not change forward predictions.

Require evidence maps and predicate/lane support lineage.

Write `axis_flow_audit.json`.

---

## 15. Contextual reason-target gate

Require a separate frozen contextual BERT encoder.

Reject:

- static word embedding mean as final target;
- trainable target encoder;
- saved embedding cache;
- GT reason input at inference.

Verify:

```text
action/reason contextual embeddings
pair projection
rank-16 train-only action basis
global projection
normalized residual target [B,768]
```

No test sample may update covariance/basis.

Action cosine after residualization must decrease on a deterministic corpus.

Write `reason_target_audit.json`.

---

## 16. PU supervision gate

Verify YAML rule loading.

Synthetic phrases must produce correct positive/contradiction/unknown masks.

Require:

```text
epochs 0–2 unknown weight = 0
later unknown regularizer <= 0.005
unknown count normalization
```

Reject per-unknown fixed negative BCE.

Require gradients to predicate/flow logits from known labels and no gradient from unknowns in semantic recovery.

Write `pu_audit.json`.

---

## 17. Semantic reason-memory gate

Require exactly 54 tokens:

```text
32 predicate
13 semantic flow
3 lane
2 axis
3 direction
1 null
```

Require names, type IDs, axis IDs, confidence, evidence maps, and lineage.

Reject equal mean pooling. Perturb confidence/query and prove semantic state changes accordingly.

Require longitudinal and lateral masks.

Write `reason_memory_audit.json`.

---

## 18. Semantic Gradient Firewall gate

### 18.1 Scaled-gradient test

Forward outputs must be identical for scale 0, 0.075, 0.20, and 1.0.

Backward norm ratios must match within tolerance:

```text
explanation 1.00
action 0.20
speed 0.075
course 0.075
```

### 18.2 Conflict-projection test

Construct analytically conflicting semantic/control losses.

Require:

```text
pre cosine < 0
post cosine >= -1e-6
forward total value unchanged
correction gradient nonzero
other task gradients preserved
zero-norm safe
```

On a real batch write conflict rate and norms.

Write `semantic_firewall_audit.json`.

---

## 19. Temporal SECA gate

Locate actual BERT captioning source with `inspect`.

Required order:

```text
multimodal BERT hidden
→ split text/image
→ action/explanation reason readers
→ recombine unchanged image hidden
→ LM head
```

Verify:

- image hidden max diff zero;
- entmax over 54 tokens;
- null token;
- separate action/explanation readers and gates;
- Xavier output projections;
- zero gate exact ADAPT;
- gate gradients nonzero at initialization;
- post-step Q/K/V/out gradients nonzero;
- action/reason generation uses correct reader under sep-cap;
- no GT explanation at inference.

Write `seca_audit.json`.

---

## 20. Axis-aware control gate

Require:

```text
base from ADAPT motion transformer
speed reads longitudinal memory
course reads lateral memory
separate attention/gates/delta heads
train-sigma residual scaling
course circular wrap
zero gate exact base
```

Synthetic targeted cases must show expected attention mass.

Reject direct flow/lane scalar input to delta MLP.

Write `control_mediation_audit.json`.

---

## 21. Prefix-future gate

Require first 24 reasoning grids only.

Final 8 visual grids and controls cannot enter prefix state.

Use tensor sentinels to prove no leak.

Temporal corruption must worsen or alter prediction in the deterministic mechanism test.

Write `prefix_future_audit.json`.

---

## 22. HardPair gate

Require start epoch 8.

Eligible pair needs:

```text
action similarity
reason dissimilarity
explicit predicate/flow contradiction
```

Verify queue/max-pair caps, inactive reasons, and gradient budget <=5% of semantic gradient.

Reject text-distance-only negatives.

Write `hardpair_audit.json`.

---

## 23. Loss and gradient gate

Require raw/weighted/gradient reports for every configured nonzero loss.

Verify normalized Huber for speed/course and circular course error.

Require each weight changes total and intended gradient.

Reject:

- double-counted text/control;
- constant-zero configured losses;
- full-video input to prefix loss;
- detached reason semantic prediction;
- test-derived training statistics.

Write `loss_audit.json`.

---

## 24. Stage-controller gate

Run mini stage transitions.

Require exact trainable sets for:

```text
Semantic Recovery epochs 0–2
Axis-Aware Motion epochs 3–7
Conflict-Aware Joint epochs 8–12
Explanation SCST epochs 13–14
```

Verify:

- control losses off in Stage R;
- ADAPT motion frozen in Stage M;
- BERT/Swin/motion partial unfreeze in Stage J;
- HardPair starts in Stage J;
- unknown schedule changes;
- all visual/control/state modules frozen in Stage S;
- SCST only trains allowed language parameters.

Reject YAML-only schedules.

Write `stage_execution_audit.json`.

---

## 25. Optimizer and scheduler gate

Create parameter-to-group manifest.

Every trainable parameter exactly once. No frozen target BERT in optimizer.

Require configured LRs and zero weight decay for gates/norm/bias.

Scheduler:

- real instance;
- warmup/cosine per stage;
- one step per optimizer step;
- state saved;
- resume produces identical next LR;
- stage multiplier applied.

Write `optimizer_scheduler_audit.json`.

---

## 26. SCST gate

Require real sampled tokens and corresponding decoder log probabilities.

Verify:

```text
sampled explanation
greedy baseline
CIDEr/METEOR reward
hallucination penalty
advantage
REINFORCE loss
25% batch routing
mixed CE objective
```

Changing token log probabilities must change SCST gradient.

Reject replay/fabricated phrase scores or reward-only logging.

Write `scst_audit.json`.

---

## 27. Checkpoint migration gate

Test migration from:

```text
released ADAPT
one valid complete V1 checkpoint
```

Require explicit loaded/missing/unexpected report.

Only V2-prefix missing keys allowed.

Reject `.tmp`, partial files, silent baseline mismatch, or optimizer restore into incompatible shapes.

Write `checkpoint_migration_audit.json`.

---

## 28. Sequence-CalAlign gate

Require deterministic 10% train-calib IDs.

Fit all six values:

```text
alpha_action
alpha_explanation
alpha_speed
alpha_course
temperature_action
temperature_explanation
```

Every alpha grid includes zero.

Main weights frozen; base/enhanced detached.

No test ID or label may enter fit.

Course interpolation uses circular residual.

Write `sequence_calalign_audit.json`.

---

## 29. Test-only evaluation gate

Require:

```text
no validation DataLoader
full test after each epoch only
best split = test
protocol tag = test_selected_user_requested
```

Test evaluation cannot mutate:

```text
model weights
optimizer
scheduler
HardPair queue
PU state
action covariance/basis
calibration parameters
```

Write before/after state hashes.

Write `test_protocol_audit.json`.

---

## 30. Best-selector gate

Run synthetic metric records.

Require separate best files and the exact lexicographic control-safe selector.

Measured ADAPT baseline, not a guessed hardcoded value, defines tolerance.

Reject arbitrary mixed-scale scalar hidden from report.

Write `best_selection_audit.json`.

---

## 31. Intervention gate

For each intervention require downstream recomputation:

```text
all flow off
longitudinal flow off
lateral flow off
factor off
predicate off
evidence tube off
random equal mass
shuffle/reverse/last frame/prefix
```

Evidence deletion must modify fused-grid evidence then rerun predicate→flow→reason→text/control.

Equal-mass random must match cumulative mass or patch count exactly.

Require real teacher-forced, generated-text, speed, and course deltas.

Write `intervention_audit.json`.

---

## 32. Traffic-flow relevance gate

Require normalized speed/course effects and conditional subsets.

Require direction-consistency metric for lateral scenes.

Reject the conclusion "course has no effect" based only on an all-test average.

Require bootstrap CI and paired permutation metadata.

Write `traffic_relevance_audit.json`.

---

## 33. Visualization and atlas gate

Real sample must produce:

```text
8-frame predicate overlay
predicate trajectories
lane-flow ribbons
flow factor/axis/direction curves
hierarchical support graph
token-to-reason panel
speed intervention panel
course intervention panel
baseline/enhanced/counterfactual text
source JSON with SHA/lineage
```

Reject one grayscale map, manual boxes, fabricated values, or JSON-only atlas.

Atlas requires grouped prototypes, failure cases, intervention distributions, standalone HTML, and links.

Write `visual_artifact_index.json`.

---

## 34. Baseline fallback gate

With all V2 deployment alphas/gates zero on identical real samples:

```text
teacher-forced text logits max diff <= 1e-4 BF16
generated action token IDs exact
generated explanation token IDs exact
speed/course max diff <= 1e-4 BF16
```

The V2 modules must execute.

Write `fallback_equivalence.json`.

---

## 35. Real-data mechanism gates

### Gate B: 8-step smoke

Require direct images, forward/backward, optimizer/scheduler, test smoke, checkpoint, no cache, finite values.

### Gate C: gradient chain

After gates leave zero initialization, require finite nonzero gradients for all intended modules.

### Gate D: 128-sample bounded mechanism fit

Require intended losses improve and no collapse:

```text
all-null memory
constant flow factors
identical lane descriptors
speed-only attention domination
course ignoring lateral memory
```

### Gate E: temporal necessity

Shuffle/reverse/last-frame-only must affect phase/trend/prefix prediction on real samples.

### Gate F: real intervention

Evidence deletion must beat equal-mass random on an intended branch, and flow-off must change outputs.

Write one JSON per gate.

---

## 36. Precision and nonfinite gate

Prefer BF16:

```text
autocast bfloat16
no GradScaler
```

If fallback FP16:

```text
GradScaler
initial scale <=4096
overflow counter
finite guard
```

Reject unexplained NaN/Inf warnings, skipped steps, or nonfinite logged values.

Write `precision_audit.json`.

---

## 37. Memory gate

Probe all configured candidates with:

```text
3 warmup
30 measured full steps
direct images
all formal losses
```

Require:

```text
no OOM
no NaN/Inf
no skipped optimizer step
peak reserved <=44.5 GiB
largest stable candidate selected
```

Report honest preferred 30–42 GiB range. No dummy allocation.

Probe SCST separately.

Write `memory_probe_selection.json`.

---

## 38. Foreground-supervisor gate

Static scan forbids:

```text
Start-Process
Start-Job
schtasks
nohup
shell &
DETACHED_PROCESS
hidden flags
```

Dynamic smoke requires:

- attached child;
- live stdout/stderr;
- heartbeat during silent child;
- pass-file SHA and remote SHA check;
- epoch artifact checks;
- no metric early stop;
- atomic latest resume;
- `.tmp` rejection;
- OOM fallback;
- explicit user stop sentinel;
- recovery workflow invalidating/reissuing audit pass.

Write `supervisor_audit.json`.

---

## 39. Required report files

The preflight directory must contain at least:

```text
git_provenance.json
formal_import_graph.json
config_binding_report.json
file_access_audit.json
adapt_text_contract.json
video_backbone_report.json
adapt_motion_equivalence.json
transport_audit.json
predicate_trajectory_audit.json
dynamic_descriptor_audit.json
lane_flow_audit.json
axis_flow_audit.json
reason_target_audit.json
pu_audit.json
reason_memory_audit.json
semantic_firewall_audit.json
seca_audit.json
control_mediation_audit.json
prefix_future_audit.json
hardpair_audit.json
loss_audit.json
stage_execution_audit.json
optimizer_scheduler_audit.json
scst_audit.json
checkpoint_migration_audit.json
sequence_calalign_audit.json
test_protocol_audit.json
best_selection_audit.json
intervention_audit.json
traffic_relevance_audit.json
visual_artifact_index.json
fallback_equivalence.json
gate_b_direct_image_8step.json
gate_c_gradient_chain.json
gate_d_mechanism_fit_128.json
gate_e_temporal_necessity.json
gate_f_real_intervention.json
precision_audit.json
memory_probe_selection.json
supervisor_audit.json
review_report.json
implementation_manifest.json
```

A report saying "checked by unit tests" without command, exit code, and artifact evidence is insufficient.

---

## 40. Review-pass contents

Only on full pass write:

```text
REVIEW_PASS_ACPR_FLOWCAL_V2.txt
```

It must contain:

```text
timestamp
reviewer role
worktree
branch
local Git SHA
GitHub SHA
clean status
plan hash
skill hash
config hash
implementation manifest hash
all test commands and exit codes
formal import graph hash
direct-image/no-cache proof
ADAPT text contract
Video-Swin and fc load proof
ADAPT motion equivalence
target-independence proof
transport warp proof
32 predicate names
lane-flow proof
13 flow names plus axis/direction
contextual target/basis proof
PU schedule proof
54-token reason memory proof
gradient scaling and conflict projection proof
SECA integration/fallback proof
axis-aware control proof
stage execution proof
optimizer/scheduler proof
SCST proof
Sequence-CalAlign no-test-fit proof
test-only evaluation proof
best selector proof
intervention/traffic relevance proof
visual/atlas proof
precision proof
memory selection
foreground proof
```

Authorization statement:

```text
ACPR_FLOWCAL_V2_IMPLEMENTATION_REVIEW_PASS
The reviewed local HEAD equals the pushed GitHub flowtrace_pmt_v1 HEAD.
The worktree is clean.
All baseline, direct-image, no-cache, transport, predicate, lane-flow,
axis-aware state, contextual reason, PU, semantic-memory, gradient-firewall,
SECA, ADAPT-motion, control, stage, scheduler, SCST, calibration,
test-only, intervention, visualization, memory, and foreground gates passed.
Formal ACPR-FlowCal++ V2 execution is authorized for this exact commit only.
```

If any gate fails:

1. delete any stale pass;
2. write exact blocker code, file, test, and evidence;
3. return to the implementer;
4. add a regression test;
5. fix, commit, push;
6. rerun the audit;
7. do not train.
