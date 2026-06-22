---
name: acpr-dynflow-implementation-audit
description: Blocking adversarial audit that authorizes the independent direct-image ACPR-DynFlow BDD-X experiment only after every architecture, metric, gradient, intervention, visualization, Git, memory, and foreground contract is dynamically proven.
---

# ACPR-DynFlow V1 Implementation Audit Skill

## 1. Purpose

This skill is the only formal training authorization gate for ACPR-DynFlow V1.

It must reject implementations that merely contain files, classes, YAML fields, loss names, or artifact schemas without executing the intended method.

Formal training is forbidden unless this skill writes:

```text
.background_runs/acpr_dynflow_v1_preflight/
REVIEW_PASS_ACPR_DYNFLOW_V1.txt
```

The pass is valid only for the exact clean local Git HEAD that equals `github/acpr_dynflow_v1`.

Any source, test, plan, config, skill, or launcher change invalidates it.

---

## 2. Required context and independent reviewer

Read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md

docs/acpr_flowcalpp/ACPR_FlowCalPP_task_plan.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_findings.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_progress.md

docs/runbooks/ACPR_DynFlow_V1_Implementation_Plan.md
docs/runbooks/ACPR_DynFlow_V1_Implementation_Manifest.json
configs/acpr_dynflow_v1_bddx_32f_224.yaml
```

Use an independent Agent B reviewer. Use Superpowers systematic-debugging, code-review, and verification-before-completion skills.

Do not accept Agent A's summary as evidence.

---

## 3. Required inputs

```text
repo_root
config_path
output_dir
device
expected_branch=acpr_dynflow_v1
expected_worktree=E:\sbw\FATE_Drive\fate_x_acpr_dynflow_v1_worktree
```

---

## 4. Git/worktree/provenance gate

Reject unless:

1. current path is the expected new worktree;
2. branch is exactly `acpr_dynflow_v1`;
3. worktree was based on recorded `flowtrace_pmt_v1` SHA;
4. `git status --porcelain` is empty;
5. `git diff --check` passes;
6. local HEAD equals:
   ```text
   git ls-remote github refs/heads/acpr_dynflow_v1
   ```
7. run artifacts, datasets, checkpoints, predictions, caches, and visuals are ignored;
8. plan, skill, config, and implementation manifest hashes are recorded.

Write:

```text
git_provenance.json
```

Reject a copied filesystem directory that is not a valid registered Git worktree.

---

## 5. Formal import-graph gate

Formal entrypoints:

```text
fate_x.engine.train_acpr_dynflow
fate_x.engine.eval_acpr_dynflow
fate_x.acpr_dynflow.model.ACPRDynFlowModel
```

Use AST and runtime import tracing.

Reject if the formal path instantiates or depends on:

```text
ACPRFlowModel
ACPRFlowCalV2Model
TokenPMTAdapter
LogSinkhornTransport
TemporalEvidenceMemory
FlowTraceLoss
old FlowCal trainers
ADAPT MultitaskVideoTransformer as the formal model
```

Shared low-level Video Swin/BERT/data/evaluator utilities are allowed.

Write:

```text
formal_import_graph.json
```

---

## 6. Model-independence gate

Scan the formal config and runtime file accesses.

Require formal training initialization only from:

```text
Kinetics Video Swin checkpoint
generic local BERT-base
user's BDD-OIA ACPR-CalAlign checkpoint
```

Reject any formal training read of:

```text
ADAPT model.bin
ADAPT task checkpoint
V1/V2 FlowCal checkpoint
cached ADAPT logits/predictions
```

A separate baseline-evaluation process may read the ADAPT checkpoint.

Write:

```text
model_independence_audit.json
```

---

## 7. Config-binding gate

Every non-comment YAML field must have:

```text
runtime consumer
audit-only consumer
or explicit documentation-only allowlist
```

Dynamically mutate representative values and prove behavior changes:

```text
number of predicates
pattern scales
lag count
control loss weight
query learning rate
text floor
memory cap
audit subset size
```

Reject orphan config.

Write:

```text
config_binding_report.json
```

---

## 8. Direct-image/no-cache gate

Trace file accesses during a real batch.

Require:

```text
frames [B,32,3,224,224]
feature_cache_enabled=false
token_cache_enabled=false
prediction_cache_enabled=false
```

Reject visual feature shards, precomputed model tokens, cached logits, or bypassed image decoding.

Train-only text thresholds/control statistics/OIA mapping are allowed and must be small, explicit artifacts.

Write:

```text
direct_image_no_cache_audit.json
```

---

## 9. ADAPT data and metric parity gate

This gate verifies comparison fairness, not model dependence.

### 9.1 Data parity

For identical sample IDs compare the ACPR-DynFlow and original ADAPT loaders:

```text
frames
input_ids
token_type_ids
masked_pos
masked_ids
raw action
raw justification
control target
valid mask
```

Required text contract:

```text
mask_prob=0.5
max_masked_tokens=45
max_seq_a_length=15
max_seq_length=30
use_sep_cap=true
```

### 9.2 Metric parity

Run the new metric bridge on existing ADAPT prediction/control artifacts and compare to the original ADAPT evaluation JSON.

Require numerical equality within `1e-6` for scalar metrics, except known library-version formatting differences documented with evidence.

Reject:

- heuristic text-decision proxy as primary;
- silent circular course replacement;
- signal order mismatch;
- different invalid-value filtering;
- different sample-ID set.

Write:

```text
adapt_metric_parity.json
signal_contract_audit.json
```

---

## 10. Typed-output gate

Run one batch and verify all dataclass fields, shapes, dtypes, devices, and finite values.

Reject positional output parsing such as:

```python
outputs[-2]
outputs[-3]
```

Write:

```text
tensor_contracts.json
```

---

## 11. Backbone gate

Verify:

- Kinetics weights loaded;
- no ADAPT task weights;
- one Video Swin forward per model forward;
- local/coarse/global outputs exist;
- local spatial resolution exceeds coarse;
- dimensions inferred;
- stage 0–1 frozen;
- stage 2–3 trainable;
- gradients reach trainable stages;
- frozen stages receive no gradients.

Write:

```text
backbone_audit.json
```

---

## 12. OIA predicate-transfer gate

Require:

- exact source branch/checkpoint metadata;
- source checkpoint SHA256;
- exact 32 predicate names/order;
- query/prototype keys loaded;
- explicit dimension mapper;
- BERT name embeddings;
- trainable residual;
- no anonymous `predicate_XX`.

Perturb each source contribution and verify initialized query changes.

Require gradient to mapper/residual and anchor loss.

Write:

```text
oia_predicate_transfer_audit.json
```

---

## 13. Dynamic predicate-field gate

Verify shapes:

```text
logits/probs/confidence [B,T,32]
tokens/query states [B,T,32,D]
evidence [B,T,32,H,W]
centroid [B,T,32,2]
relative motion [B,T-1,32,2]
lane mass [B,T,32,3]
```

Require:

- entmax exact zeros;
- evidence maps normalize;
- region priors affect expected predicates;
- recurrent query uses previous state;
- changing temporal order changes query states;
- current frame always contributes;
- common camera shift is subtracted;
- confidence is not a constant.

Reject full Sinkhorn or ignored recurrence.

Write:

```text
dynamic_predicate_audit.json
```

---

## 14. nnPU/CalAlign gate

Synthetic and real-rule tests must verify:

- positives;
- reliable contradictions;
- unknowns;
- exclusions such as `light traffic`;
- nonnegative PU risk;
- unknown is not hard negative;
- predicate-specific priors;
- train-only histogram updates;
- test evaluation cannot update calibration;
- threshold/temperature state saves and resumes.

Compare gradients with an unknown-only sample: no ordinary negative BCE gradient is allowed.

Write:

```text
nnpu_calalign_audit.json
```

---

## 15. Homogenization gate

Require raw covariates contain:

```text
presence
presence delta
centroid
relative centroid motion
left/center/right mass
confidence
```

Ablate each group and prove the homogenized output changes.

Require shared homogenizer plus semantic-name contribution.

Write:

```text
covariate_homogenization_audit.json
```

---

## 16. Multi-scale pattern-router gate

Require actual 1×/2×/4× branches and progressive coarse-to-fine fusion.

Synthetic sequences:

```text
stable
increasing
decreasing
oscillating
```

Require corresponding dominant pattern probabilities.

Temporal reverse must exchange forming/releasing behavior.

All four pattern adapters and routing projection require gradients.

Reject a single-scale implementation with decorative scale names.

Write:

```text
pattern_router_audit.json
```

---

## 17. Mesoscopic corridor-flow gate

Require distinct left/center/right masks and outputs.

Synthetic cases:

```text
center congestion
left corridor freer
right corridor blocked
queue forming
queue releasing
```

Require correct signs/order.

Participant aggregation must exclude signal/road predicates where inappropriate.

Reject copying one global statistic into all corridors.

Write:

```text
mesoscopic_flow_audit.json
```

---

## 18. Traffic-state composition gate

Require exact 13 factors and signed lateral bias.

Verify:

- factor queries read predicates, corridors, patterns, names, and grammar;
- entmax support is sparse;
- low-rank regime/phase/source interaction executes;
- factor evidence maps equal weighted predicate evidence;
- lineage is complete;
- different factor combinations produce distinct tokens.

Reject one global vector followed by 13 linear classifiers as the sole state representation.

Write:

```text
traffic_state_audit.json
```

---

## 19. Response-lag gate

Require four lags `0..3`, causal masks, and normalized weights.

Synthetic delayed-event test must recover the known lag within tolerance.

Disable-lag and learned-lag outputs must differ on delayed samples.

Require gradients to all lag projections.

Write:

```text
response_lag_audit.json
```

---

## 20. Signal-codec and global-decision gate

Require:

- signal order discovered and recorded;
- train mean/std computed from train only;
- invalid values masked;
- encode/decode round trip;
- official evaluation uses decoded raw units;
- global stream outputs `[B,32,2]`;
- no ADAPT motion head/checkpoint;
- no target values in forward.

Target-leak test:

```text
same video, different control targets
→ identical forward predictions
```

Write:

```text
signal_codec_audit.json
global_decision_audit.json
```

---

## 21. Decision-ledger gate

Require exact identity:

```text
final = global + sum(factor contributions)
```

in normalized and raw units.

Require:

- separate speed/course factor readers;
- speed prior emphasizes longitudinal factors;
- course prior emphasizes lateral/directional factors;
- contributions vary over time;
- contribution sum receives residual-target supervision;
- global and final branches both receive task supervision;
- state-off subtracts the exact intended contribution and recomputes dependent text.

Synthetic left/right cases must produce different course contribution signs.

Write:

```text
decision_ledger_audit.json
```

---

## 22. Text-decoder gate

Require generic BERT-base initialization and no ADAPT text checkpoint.

Verify:

- action and explanation token losses are separate;
- action reads global+flow context;
- explanation reads state/contribution/action context;
- global decision context is detached for explanation;
- predicate/flow path receives explanation gradients;
- global decision path does not receive explanation-only gradients;
- contribution-alignment loss uses real ledger values;
- inference receives no GT action/justification;
- generated action conditions explanation only through generated/decoder state.

Write:

```text
text_decoder_audit.json
gradient_direction_audit.json
```

---

## 23. Loss gate

Require every configured nonzero loss to:

- be a real tensor;
- contribute exactly once;
- be separately logged raw/weighted;
- give gradients to intended parameters;
- remain finite.

Reject:

```text
duplicated text loss
HardPair
SCST
uncertainty weighting
PCGrad/firewall
ADAPT distillation
unknown-negative BCE
```

Verify normalized Huber and first-difference control loss.

Write:

```text
loss_audit.json
```

---

## 24. Optimizer/scheduler gate

Build parameter-to-group manifest.

Every trainable parameter exactly once. Every frozen parameter absent.

Verify exact fixed trainability for all epochs; no stage controller.

Require:

```text
5% warm-up
cosine decay
one scheduler step per optimizer step
save/restore scheduler
save/restore optimizer
```

Resume must reproduce the next LR and optimizer step.

Write:

```text
optimizer_scheduler_audit.json
```

---

## 25. Best-selection and test-only gate

Require:

- no validation DataLoader;
- full test once after every epoch;
- separate best text/control/joint/test;
- measured ADAPT reference used;
- exact text-floor and lexicographic selector;
- test cannot update weights, optimizer, scheduler, PU histograms, or calibration state;
- no metric early stop.

Hash model/calibration state before and after test.

Write:

```text
test_protocol_audit.json
best_selector_audit.json
```

---

## 26. Real-data smoke gate

Use direct images and at least eight optimizer steps.

Require:

```text
forward/backward
all formal outputs
finite losses
optimizer and scheduler steps
atomic latest checkpoint
test smoke evaluation
no cache
```

Write:

```text
gate_real_direct_image_smoke.json
```

---

## 27. Gradient-chain gate

On a real batch, require finite nonzero gradients for every intended trainable component.

Also verify frozen components have no gradient.

Write parameter-level norms and the loss source that reaches each module.

Write:

```text
gate_gradient_chain.json
```

---

## 28. 128-sample mechanism-fit gate

On deterministic 128 training samples, run bounded updates.

Require improvements in:

```text
final speed/course loss
global speed/course loss
flow residual loss
action loss
explanation loss
known-label nnPU risk
pattern semantic loss
contribution alignment
```

Reject collapse:

```text
all predicates identical
all states constant
all routing stable
all flow contributions zero
all explanation attention on one state
course ignores directional factors
```

Write:

```text
gate_mechanism_fit_128.json
```

---

## 29. Temporal/lag gate

On real and synthetic cases require:

- shuffle/reverse changes pattern and states;
- last-frame-only differs;
- learned lag differs from lag-zero;
- known delayed response is recovered.

Write:

```text
gate_temporal_lag_necessity.json
```

---

## 30. Intervention gate

Supported:

```text
all flow off
regime/phase/source off
factor off
predicate off
evidence tube off
equal-mass random
temporal shuffle/reverse
lag zero
last-frame-only
```

Each intervention must rerun all downstream computations from the earliest affected tensor.

Require actual text/control deltas.

Equal-mass random must match mass or patch count exactly.

Write:

```text
gate_intervention.json
```

---

## 31. Visualization/atlas gate

One real sample must produce the complete Dynamic Traffic Decision Ledger:

```text
predicate tubes
corridor ribbons
pattern/state lattice
response-lag ribbon
signed speed/course waterfall
contribution-aligned text
counterfactual twin
source JSON with tensor lineage and SHA
```

Mini atlas requires HTML and JSON index.

Reject grayscale-only heatmaps, manual boxes, fabricated deltas, or record dumps.

Write:

```text
visual_artifact_index.json
```

---

## 32. Memory gate

Probe every candidate with three warm-up and thirty measured full steps.

Require:

```text
direct images
all formal losses
BF16
no OOM
no nonfinite values
no skipped optimizer step
peak reserved <=46.0 GiB
largest stable batch selected
```

No dummy allocation.

Write:

```text
memory_probe_selection.json
```

---

## 33. Foreground-supervisor gate

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
- heartbeat;
- pass/current local/current remote SHA verification;
- epoch artifact checks;
- no metric stop;
- transient retry;
- OOM fallback;
- atomic latest resume;
- `.tmp` rejection;
- explicit user-stop sentinel;
- code-fix path that invalidates and reruns audit.

Write:

```text
foreground_supervisor_audit.json
```

---

## 34. Required audit files

The preflight directory must contain:

```text
git_provenance.json
formal_import_graph.json
model_independence_audit.json
config_binding_report.json
direct_image_no_cache_audit.json
adapt_metric_parity.json
signal_contract_audit.json
tensor_contracts.json
backbone_audit.json
oia_predicate_transfer_audit.json
dynamic_predicate_audit.json
nnpu_calalign_audit.json
covariate_homogenization_audit.json
pattern_router_audit.json
mesoscopic_flow_audit.json
traffic_state_audit.json
response_lag_audit.json
signal_codec_audit.json
global_decision_audit.json
decision_ledger_audit.json
text_decoder_audit.json
gradient_direction_audit.json
loss_audit.json
optimizer_scheduler_audit.json
test_protocol_audit.json
best_selector_audit.json
gate_real_direct_image_smoke.json
gate_gradient_chain.json
gate_mechanism_fit_128.json
gate_temporal_lag_necessity.json
gate_intervention.json
visual_artifact_index.json
memory_probe_selection.json
foreground_supervisor_audit.json
review_report.json
implementation_manifest.json
```

---

## 35. Review pass

Only on full pass write:

```text
REVIEW_PASS_ACPR_DYNFLOW_V1.txt
```

It must contain:

```text
timestamp
independent reviewer role
source worktree/branch/SHA
new worktree/branch/SHA
GitHub SHA
clean status
plan/config/skill/manifest hashes
all commands and exit codes
ADAPT metric parity proof
model independence proof
direct-image/no-cache proof
OIA predicate transfer proof
32 predicate names
nnPU/CalAlign proof
temporal/pattern/lag proof
decision-ledger identity
text gradient-direction proof
test-only proof
intervention proof
visual proof
memory selection
foreground proof
```

Authorization statement:

```text
ACPR_DYNFLOW_V1_IMPLEMENTATION_REVIEW_PASS
The reviewed local HEAD equals the pushed GitHub acpr_dynflow_v1 HEAD.
The worktree is clean.
The independent ACPR-DynFlow model, ADAPT-compatible evaluation, direct-image
training, OIA predicate transfer, calibrated nnPU, dynamic covariates,
multi-scale pattern routing, compositional traffic states, response lag,
factor-decomposed decision ledger, contribution-aligned text, interventions,
visualization, memory policy, and foreground supervisor all passed.
Formal execution is authorized for this exact commit only.
```

If any gate fails:

1. remove any stale pass;
2. write exact blockers;
3. return to Agent A;
4. add regression tests;
5. fix, commit, push;
6. rerun the complete affected audit and final full audit;
7. do not train.
