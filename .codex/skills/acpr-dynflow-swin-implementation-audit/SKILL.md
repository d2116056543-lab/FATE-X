---
name: acpr-dynflow-swin-implementation-audit
description: Blocking adversarial audit for the independent full-capacity ACPR-DynFlow-Swin BDD-X implementation. It authorizes training only after architecture, metric parity, BF16 throughput, gradients, exact ledger, interventions, visualization, Git, and foreground execution are dynamically proven.
---

# ACPR-DynFlow-Swin V1 Implementation Audit Skill

## 1. Authority

This skill is the only formal authorization gate for ACPR-DynFlow-Swin V1.

It must reject:

- files/classes that exist but are unreachable;
- YAML fields without runtime consumers;
- legacy DynFlow/FlowCal imports;
- placeholder reports;
- fake BF16 or memory probes;
- ADAPT task-weight dependence;
- disconnected predicate/pattern/lag paths;
- wrong loss signs;
- target leakage;
- duplicated speed/course or action/explanation losses;
- post-hoc decision contributions;
- non-autoregressive text evaluation;
- fake interventions;
- placeholder Canvas/Atlas;
- validation evaluation;
- test-fitted model/calibration;
- detached/background execution.

Formal training is forbidden unless this skill writes:

```text
.background_runs/acpr_dynflow_swin_v1_preflight/
REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt
```

The pass is valid only for the exact clean local HEAD equal to `github/acpr_dynflow_v1`.

Any source/config/test/plan/skill/script change invalidates it.

---

## 2. Required context and reviewer

Read in full:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md

docs/acpr_flowcalpp/ACPR_FlowCalPP_task_plan.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_findings.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_progress.md

docs/runbooks/ACPR_DynFlow_Swin_V1_Implementation_Plan.md
docs/runbooks/ACPR_DynFlow_Swin_V1_Implementation_Manifest.json
configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml
```

Use an independent Agent B reviewer and Superpowers code-review, systematic-debugging, and verification-before-completion skills.

Do not trust Agent A's checklist without dynamic evidence.

---

## 3. Git/worktree gate

Expected:

```text
worktree = E:\sbw\FATE_Drive\fate_x_acpr_dynflow_v1_worktree
branch = acpr_dynflow_v1
remote = github
```

Reject unless:

1. exact worktree;
2. exact branch;
3. `git status --porcelain` empty;
4. `git diff --check` passes;
5. local HEAD equals `git ls-remote github refs/heads/acpr_dynflow_v1`;
6. run artifacts/data/checkpoints/predictions/visuals/caches are ignored;
7. plan/config/skill/manifest hashes are recorded.

Write `git_provenance.json`.

---

## 4. Formal import graph

Formal entrypoints:

```text
fate_x.engine.train_acpr_dynflow_swin
fate_x.engine.eval_acpr_dynflow_swin
fate_x.acpr_dynflow_swin.model.ACPRDynFlowSwinModel
```

Trace imports with AST and runtime `inspect`.

Reject reachable imports/instances of:

```text
fate_x.acpr_dynflow.model.ACPRDynFlowModel
ACPRFlowModel
ACPRFlowCalV2Model
FlowTracePMTModel
TokenPMTAdapter
LogSinkhornTransport
legacy FlowCal/DynFlow trainers
```

Shared low-level Video Swin/BERT/data/evaluator utilities are allowed.

Write `formal_import_graph.json`.

---

## 5. Config-binding gate

For every non-comment YAML field establish:

```text
runtime consumer
audit-only consumer
or explicit documentation-only allowlist
```

Dynamically mutate and prove behavior changes for:

```text
semantic slot count
pattern dilations
lag count
benefit-gate margin
learning rates
text floor
memory cap
audit sample count
```

Reject orphan config.

Write `config_binding_report.json`.

---

## 6. Model-independence gate

Trace all file reads during model construction and real smoke.

Formal model may read only:

```text
Kinetics-600 Video Swin checkpoint
generic BERT-base files
BDD-OIA ACPR query checkpoint
predicate/rule/grammar YAML
```

Reject reads of:

```text
ADAPT task checkpoint
FlowTrace/FlowCal/DynFlow checkpoint
cached ADAPT predictions/logits/features
```

The separate baseline evaluator may read ADAPT.

Write `model_independence_audit.json`.

---

## 7. Direct-image/no-cache gate

Require real:

```text
frames [B,32,3,224,224]
```

Trace file access.

Reject:

```text
visual feature shards
precomputed Video Swin/DINO tokens
token caches
model-logit caches
bypassed frame decoding
```

Small train statistics are allowed and documented.

Write `direct_image_no_cache_audit.json`.

---

## 8. ADAPT data/metric parity gate

### Data parity

For identical test sample IDs compare formal and ADAPT loaders:

```text
frames
input IDs
token types
masked positions/IDs
attention-mask semantics
raw action
raw justification
control target
valid mask
```

### Metric parity

Run the new metric bridge on existing ADAPT predictions/control outputs and reproduce the original evaluation JSON within `1e-6`, except documented library formatting.

Reject:

- heuristic text-decision proxy as primary;
- signal-order mismatch;
- different invalid mask;
- silent circular metric replacement;
- different sample set.

Write:

```text
adapt_metric_parity.json
signal_contract_audit.json
```

---

## 9. Typed-output gate

Run one real batch and verify every dataclass field, shape, dtype, device, and finite status.

Reject positional tail parsing.

Write `tensor_contracts.json`.

---

## 10. Video Swin-B gate

Require formal use of repository ADAPT-proven Video Swin-B, initialized from Kinetics-600.

Verify:

- no `torchvision.swin3d_b` in formal import graph;
- exactly one forward;
- 32×224 input;
- native middle and final stages;
- middle stage is not interpolated final stage;
- no forced `.float()`;
- BF16 tensors observed under autocast;
- all Video Swin parameters trainable;
- backbone LR group correct;
- gradients finite/nonzero.

Write `video_swin_backbone_audit.json`.

---

## 11. OIA transfer gate

Require:

- exact 32 names/order;
- source branch/checkpoint path and SHA256;
- learned OIA query key loaded;
- true BERT name embedding;
- explicit mapper;
- per-predicate transfer gate initialized near configured value;
- trainable residual;
- anchor loss.

Perturb OIA query, name embedding, and residual independently and prove query output changes.

Reject hash-byte pseudo-name embeddings.

Write `oia_transfer_audit.json`.

---

## 12. Dynamic predicate gate

Shapes:

```text
query/token [B,T,32,D]
logits/probs/confidence [B,T,32]
evidence [B,T,32,H,W]
centroid [B,T,32,2]
relative motion [B,T-1,32,2]
corridor mass [B,T,32,3]
```

Require:

- entmax exact zeros;
- evidence normalization;
- region-prior effect;
- recurrent dependence on previous predicate state;
- temporal order changes queries;
- current frame contributes;
- ego shift subtracted;
- confidence nonconstant;
- all intended gradients.

Write `dynamic_predicate_audit.json`.

---

## 13. nnPU/CalAlign gate

Require all 32 rule entries and explicit supervision mode.

Synthetic text tests include:

```text
traffic light red
light traffic
cars ahead stopped
road is clear
left lane blocked/open
pedestrian crossing
```

Verify:

- positive;
- reliable negative;
- exclusion;
- unlabeled;
- nonnegative risk;
- unknown-only sample has no ordinary negative-BCE gradient;
- train-only prior/threshold/temperature updates;
- test cannot mutate calibration;
- save/resume exact.

Real smoke must have nonzero positive/reliable-negative counts across the batch window.

Write `nnpu_calalign_audit.json`.

---

## 14. Mass-preserving consolidation gate

Require exactly five slots and 80 tokens for the expected 16 native time units.

Verify:

```text
assignment sum over slots = 1
mass nonnegative
residual slot present
weighted-token conservation error <=1e-5 FP32 / 1e-3 BF16
provenance maps to original 7×7 tokens
```

Ablate predicate priors and prove semantic assignments change while residual preserves unassigned mass.

Reject generic top-k or mean pooling presented as the formal consolidator.

Write `semantic_consolidation_audit.json`.

---

## 15. Corridor-flow gate

Require distinct left/center/right tensors.

Participant predicates only contribute to occupancy/mobility counts.

Synthetic cases:

```text
center queue
left freer
right blocked
queue forming
queue releasing
```

Require correct differences and temporal sensitivity.

Reject replicated global statistics.

Write `corridor_flow_audit.json`.

---

## 16. Pattern–traffic gate

Require actual dilation-1/2/4 branches and progressive fusion.

Synthetic:

```text
stable
increasing
decreasing
oscillating
```

Require dominant expected pattern.

Trace graph to prove pattern output affects factor tokens and final control.

Require exact 13 factor names, separate regime/phase/source activations, entmax supports, evidence maps, and lineage.

Reject:

```text
pattern logits squared toward zero
mean factor probability penalty
one global vector + decorative 13 classifiers
```

Write `pattern_traffic_audit.json`.

---

## 17. Response-lag gate

Require lags `0..3`, causal mask, normalized weights.

Synthetic delayed-event test must recover known lag.

Disabling lag must change factor tokens and decisions on delayed cases.

Require gradients to lag parameters.

Write `response_lag_audit.json`.

---

## 18. Query motion-transformer gate

Require:

```text
BERT-base hidden size/heads/FFN
12 encoder layers
32 output-time queries
16 global temporal tokens
80 semantic tokens
```

No ADAPT sensor weights.

Target leakage:

```text
same video, same frame count, different control targets
→ identical forward prediction
```

Require independent global prediction and gradients.

Write `query_motion_transformer_audit.json`.

---

## 19. Decision-ledger gate

Require separate speed/course readers.

Exact identities in normalized/raw units:

```text
gated contribution = benefit gate × raw contribution
final = global + sum(gated contributions)
```

Require:

- residual target is `GT-global` with stop-gradient;
- benefit target uses detached actual improvement;
- safe hinge penalizes only harmful flow changes;
- no contribution-magnitude penalty substituted for residual objective;
- speed/course losses separate by signal name;
- target-difference loss, not prediction-flatness loss;
- left/right synthetic cases change course contribution sign.

Write `decision_ledger_audit.json`.

---

## 20. Autoregressive text gate

Locate actual BERT captioning decoder and generation path with `inspect`.

Require:

- generic BERT-base initialization;
- no ADAPT task text weights;
- dynamic visual attention mask for semantic/factor/contribution tokens;
- autoregressive action generation;
- autoregressive explanation generation after action;
- separate masked action/explanation losses;
- contribution-reason adapter before LM head;
- image hidden unchanged by adapter;
- explanation global context detached;
- explanation gradients still reach predicate/traffic path;
- no GT action/justification at inference;
- beam/length/tokenizer parity with ADAPT.

Reject position-wise argmax over a bidirectional BERT sequence.

Write:

```text
text_decoder_audit.json
gradient_direction_audit.json
```

---

## 21. Loss gate

Require raw, weighted, and gradient-target records for every configured loss.

Verify each nonzero weight affects total exactly once.

Reject:

```text
one control loss duplicated as speed/course
one caption loss duplicated as action/explanation
contribution absolute magnitude as residual loss
prediction absolute first difference without target
factor probability mean as grammar loss
pattern-logit square-to-zero objective
unknown-negative BCE
HardPair/SCST/PCGrad in formal run
```

Write `loss_audit.json`.

---

## 22. Optimizer/precision/scheduler gate

Parameter-to-group manifest:

- every trainable parameter exactly once;
- correct LR and weight decay;
- no calibration accumulator/rule target in optimizer.

Runtime:

```text
native CUDA BF16 autocast
no backbone forced FP32
gradient clip
10% warm-up
linear decay
one scheduler step per optimizer step
save/resume exact
```

Resume must reproduce next LR and optimizer state.

Write:

```text
optimizer_precision_scheduler_audit.json
```

---

## 23. Throughput/memory gate

The probe must execute 10 warm-up + 100 measured real steps for every viable candidate.

Record component timings, samples/s, peak memory, projected hours/epoch.

Require:

```text
peak reserved <=44 GiB
preferred 36–42 GiB where throughput-optimal
projected train epoch <=4 hours
data time <=20%
no OOM
no NaN/Inf
no skipped optimizer step
```

Select highest samples/s, not highest memory use.

Reject static/fabricated memory JSON.

Write `throughput_memory_probe.json`.

---

## 24. Trainer/test-only gate

Require one fixed 16-epoch module policy; no stage controller.

Full test after every epoch.

No validation DataLoader.

Test cannot mutate:

```text
model
optimizer
scheduler
nnPU priors/histograms
CalAlign thresholds/temperatures
RNG-dependent train state
```

Hash states before/after test.

No metric early stop.

Write `test_protocol_audit.json`.

---

## 25. Best-selector gate

Use synthetic records to verify:

- best text;
- normalized best control vs measured ADAPT;
- decision-first text-safe best-test selector;
- exact 0.85 text floor;
- full comparison tuple written.

Reject hidden arbitrary mixed scalar.

Write `best_selector_audit.json`.

---

## 26. Real smoke and mechanism gates

### Real 8-step direct-image smoke

Require actual forward/backward/optimizer/scheduler/checkpoint/test generation/control metrics.

### Gradient chain

Finite nonzero gradients for all intended trainable modules.

### 128-sample bounded fit

Require intended losses improve and reject predicate/pattern/factor/contribution collapse.

### Identity tests

Mass conservation and decision ledger exact reconstruction.

### Temporal tests

Shuffle/reverse/lag-zero/last-frame affect intended states and decisions.

Write one JSON per gate.

---

## 27. Intervention gate

Supported:

```text
all-flow/global-only
regime/phase/source off
factor off
predicate off
evidence tube off
equal-mass random
shuffle/reverse
lag zero
last-frame-only
residual-slot-only
```

Every intervention reruns from earliest affected layer.

Require actual generated text and control changes.

Equal-mass random must match evidence mass or patch count.

Ledger factor-off effect must equal removal/recomputation of the exact contribution when no upstream state changes are requested.

Write `intervention_audit.json`.

---

## 28. Visualization gate

One real case must contain:

```text
predicate tubes
semantic consolidation/provenance
corridor ribbons
pattern/state lattice
lag ribbon
exact signed decision waterfall
benefit gate
contribution-aligned generated text
counterfactual comparisons
source JSON with tensor lineage and Git/config hashes
```

Mini atlas must be real standalone HTML + JSON.

Reject grayscale-only output, manual boxes, fabricated numbers, or record dump.

Write `visual_artifact_index.json`.

---

## 29. Foreground supervisor gate

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
- 60-second heartbeat;
- review/local/remote SHA verification;
- epoch artifact checks;
- no metric stop;
- retry transient failures;
- OOM fallback;
- atomic latest resume;
- `.tmp` rejection;
- explicit user stop sentinel;
- code-fix workflow invalidating/reissuing review pass.

Write `foreground_supervisor_audit.json`.

---

## 30. Required reports

Preflight directory must contain:

```text
git_provenance.json
formal_import_graph.json
config_binding_report.json
model_independence_audit.json
direct_image_no_cache_audit.json
adapt_metric_parity.json
signal_contract_audit.json
tensor_contracts.json
video_swin_backbone_audit.json
oia_transfer_audit.json
dynamic_predicate_audit.json
nnpu_calalign_audit.json
semantic_consolidation_audit.json
corridor_flow_audit.json
pattern_traffic_audit.json
response_lag_audit.json
query_motion_transformer_audit.json
decision_ledger_audit.json
text_decoder_audit.json
gradient_direction_audit.json
loss_audit.json
optimizer_precision_scheduler_audit.json
throughput_memory_probe.json
test_protocol_audit.json
best_selector_audit.json
gate_real_direct_image_smoke.json
gate_gradient_chain.json
gate_mechanism_fit_128.json
gate_identity_checks.json
gate_temporal_lag.json
intervention_audit.json
visual_artifact_index.json
foreground_supervisor_audit.json
implementation_manifest.json
review_report.json
```

A report stating only “covered by tests” is insufficient. Include commands, exit codes, tensor stats, artifact paths, and SHA256.

---

## 31. Review pass

Only on full pass write:

```text
REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt
```

Contents:

```text
timestamp
independent reviewer
worktree
branch
local/GitHub SHA
clean status
plan/config/skill/manifest hashes
all commands and exit codes
metric parity
model independence
direct-image/no-cache
Video Swin native/BF16/one-forward proof
OIA transfer and 32 names
nnPU positive/reliable-negative proof
mass conservation
pattern-to-state connectivity
lag proof
motion target independence
ledger exact identity
benefit/safe-loss direction
autoregressive text proof
separate loss proof
gradient direction
test-only proof
intervention proof
visual proof
selected batch/accumulation
peak memory
samples/s
projected epoch time
foreground proof
```

Authorization statement:

```text
ACPR_DYNFLOW_SWIN_V1_IMPLEMENTATION_REVIEW_PASS
The reviewed local HEAD equals the pushed GitHub acpr_dynflow_v1 HEAD.
The worktree is clean.
The independent full-capacity Video Swin-B ACPR-DynFlow-Swin implementation,
ADAPT-compatible metrics, OIA predicate transfer, calibrated nnPU,
mass-preserving semantic consolidation, pattern-lag traffic reasoning,
query motion transformer, exact benefit-constrained decision ledger,
autoregressive contribution-aligned text generation, interventions,
visualization, BF16 throughput, memory, and foreground execution all passed.
Formal execution is authorized for this exact commit only.
```

On any failure:

1. delete stale pass;
2. write exact blocker;
3. return to Agent A;
4. add regression test;
5. fix, commit, push;
6. rerun affected gates and final full audit;
7. do not train.
