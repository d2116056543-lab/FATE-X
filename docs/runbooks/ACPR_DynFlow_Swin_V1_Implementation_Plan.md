# ACPR-DynFlow-Swin V1
## Codex code-level implementation, audit, experiment, and foreground-supervision contract

**Repository:** `https://github.com/d2116056543-lab/FATE-X`
**Current worktree:** `E:\sbw\FATE_Drive\fate_x_acpr_dynflow_v1_worktree`
**Current branch:** `acpr_dynflow_v1`
**Expected starting branch HEAD when this contract was written:** `7da4f84d40beb6908b1f34f0b134709317d5a0e4`
**No new worktree. No new branch. Modify the current worktree and push the same branch.**
**Formal method:** `ACPR-DynFlow-Swin V1`
**Formal config:** `configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml`
**Formal namespace:** `fate_x.acpr_dynflow_swin`
**Formal trainer:** `python -m fate_x.engine.train_acpr_dynflow_swin`

---

# 0. Formal objective

Implement an independent, end-to-end BDD-X model that retains ADAPT-level model capacity but does not load or depend on an ADAPT task checkpoint.

The model extends the user's BDD-OIA ACPR line from static image reasoning to dynamic video reasoning:

```text
32 direct RGB frames
→ one full-capacity Video Swin-B forward
→ 32 transferred named ACPR predicate trajectories
→ calibrated positive–unlabeled predicate semantics
→ predicate-guided mass-preserving semantic token consolidation
→ pattern–lag mesoscopic traffic states
→ independent full-capacity motion transformer
→ benefit-constrained exact speed/course decision ledger
→ ADAPT-compatible autoregressive action/explanation generation
→ model-level traffic-flow influence verification
```

The method must remain independent:

```text
generic Kinetics-600 Video Swin-B initialization
+ generic BERT-base initialization
+ the user's BDD-OIA ACPR predicate-query checkpoint
```

Formal ACPR-DynFlow-Swin training must not load:

```text
ADAPT task checkpoint
FlowTrace checkpoint
FlowCal V1/V2 checkpoint
current ACPR-DynFlow V1 checkpoint
cached ADAPT predictions/logits/features
```

ADAPT is used only as:

1. an external comparison baseline;
2. the source of the official BDD-X metric protocol;
3. a reusable public architecture/code reference.

---

# 1. Current-branch defects that the new formal path must isolate

The current `fate_x.acpr_dynflow` path may remain for historical diagnosis, but it must not be imported by the new formal trainer.

The new implementation must explicitly eliminate these current defects:

1. `positive=0`, `reliable_negative=0`, `unlabeled=1` hard-coded for every predicate.
2. Pattern-router output computed but not consumed by the traffic reasoner.
3. One control loss duplicated as both speed and course loss.
4. Flow-residual losses implemented as contribution magnitude penalties, pushing the traffic contribution toward zero.
5. Control first-difference loss implemented as `|prediction[t]-prediction[t-1]|`, pushing motion toward a constant sequence instead of matching target dynamics.
6. Traffic grammar implemented as mean probability minimization.
7. Generic `BertModel + position-wise argmax` used instead of the ADAPT-compatible autoregressive decoder.
8. `torchvision.swin3d_b`, forced FP32 inputs, and interpolated final features used instead of the repository's ADAPT-proven Video Swin-B path and native intermediate stages.
9. One global optimizer LR despite a config containing module-specific learning rates.
10. BF16 and gradient-checkpointing configuration not bound to runtime.
11. Memory probe writing a report without executing real forward/backward measurements.
12. Preflight writing placeholder `passed=true` reports.
13. Renderer/Atlas producing placeholder JSON/HTML rather than real tensor-linked visualizations.
14. Formal evaluation using direct argmax rather than ADAPT-compatible generation.
15. Interventions zeroing display tensors or whole branches without recomputing the earliest affected downstream path.

The new formal namespace prevents legacy imports from silently surviving.

---

# 2. Non-negotiable research invariants

## 2.1 Model capacity

- Video backbone: full Video Swin-B, 32 frames, 224×224.
- Text decoder: the repository's ADAPT-compatible autoregressive BERT captioning architecture, initialized from generic BERT-base.
- Motion head: BERT-base-capacity query motion transformer, independently initialized.
- No lightweight substitute for the main Video Swin-B, caption decoder, or motion transformer.

Efficiency must come from using semantic information efficiently after the backbone, not from reducing the backbone class.

## 2.2 Single backbone pass

Every forward performs exactly one Video Swin-B pass.

The same pass supplies:

```text
native middle-stage feature grid for predicates
native final-stage dense grid for semantic consolidation
temporally pooled final tokens for global motion
```

No second Video Swin pass for text, control, predicates, prefix, intervention diagnostics, or auxiliary losses.

## 2.3 Exact ACPR continuity

The formal model contains exactly the 32 predicates from the user's BDD-OIA ACPR-CalAlign V1.2 ontology and loads the corresponding learned query/prototype tensor.

Anonymous predicate slots are forbidden.

## 2.4 Exact decision decomposition

For every sample and timestep:

\[
U^{final}_{t}
=
U^{global}_{t}
+
\sum_{k=1}^{13}
\Delta U^{gated}_{t,k}
\]

The equality must hold in both normalized and raw signal units within numerical tolerance.

Saved factor effects are the actual tensors added in the forward pass, not saliency approximations.

## 2.5 Traffic-flow semantics

The model outputs:

```text
Regime:
  clear_open_flow
  stable_following
  dense_following
  queue_congestion

Phase:
  forming
  stable
  releasing
  oscillating

Source:
  traffic_signal
  lead_vehicle_group
  merge_lane_constraint
  turn_intersection
  vulnerable_obstacle_conflict

Additional:
  signed lateral bias
  response-lag distribution
```

These are described as **ego-centric mesoscopic visual traffic states**, not physical road-network density or metric traffic flow.

## 2.6 Direct-image training

Formal input:

```text
[B, 32, 3, 224, 224]
```

No learned image/video feature cache, token cache, model-logit cache, or ADAPT prediction cache.

Allowed small train-only artifacts:

```text
control statistics
predicate class priors
predicate calibration histograms
OIA query-transfer metadata
fixed test-audit sample IDs
```

## 2.7 One fixed training regime

There is one 16-epoch end-to-end run.

No semantic/motion/joint/SCST stage schedule. The module set and `requires_grad` policy remain fixed throughout the run.

Ordinary learning-rate warm-up and contribution-scale warm-up are allowed.

## 2.8 User-requested evaluation protocol

- full test evaluation after every epoch;
- no formal validation loader;
- test-selected best checkpoints;
- protocol recorded as `test_selected_user_requested`;
- no metric early stop.

## 2.9 Foreground execution

The supervisor remains attached to the foreground terminal and streams child output.

It must not stop because metrics decline or plateau.

---

# 3. Mandatory context and Git procedure

Before code, tests, training, evaluation, or Git changes, Codex must read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md

docs/acpr_flowcalpp/ACPR_FlowCalPP_task_plan.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_findings.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_progress.md

docs/runbooks/ACPR_DynFlow_Swin_V1_Implementation_Plan.md
docs/runbooks/ACPR_DynFlow_Swin_V1_Implementation_Manifest.json
.codex/skills/acpr-dynflow-swin-implementation-audit/SKILL.md
configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml
```

Training/experiment status is appended only to the established task/findings/progress ledgers. Runbooks and audit skills are non-status documents.

## 3.1 Current worktree only

```powershell
$Repo = "E:\sbw\FATE_Drive\fate_x_acpr_dynflow_v1_worktree"
Set-Location $Repo

git branch --show-current
git status --short
git remote -v
git fetch github
git rev-parse HEAD
git ls-remote github refs/heads/acpr_dynflow_v1
```

Required branch:

```text
acpr_dynflow_v1
```

Do not create another branch or worktree.

## 3.2 Dirty-worktree safety

If dirty:

1. inspect every modified/untracked source file;
2. save a patch under ignored:
   `.background_runs/pre_dynflow_swin_snapshot/`;
3. separate source changes from run artifacts;
4. compile/test intended source changes;
5. create a safety-snapshot commit on `acpr_dynflow_v1`;
6. push to `github/acpr_dynflow_v1`;
7. verify local and remote SHA equality.

Forbidden:

```text
git reset --hard
git clean -fd
discarding unknown edits
overwriting a concurrent process's changes
```

## 3.3 Commit discipline

Use coherent commits:

1. plan/config/skill/manifest;
2. typed contracts and formal namespace;
3. backbone and evaluator parity;
4. predicate transfer/field/nnPU;
5. token consolidation;
6. pattern–lag traffic reasoner;
7. motion transformer and decision ledger;
8. text decoder integration;
9. trainer/optimizer/precision;
10. intervention/visualization;
11. preflight/audit/supervisor;
12. independent-review fixes.

After each:

```text
targeted tests
git diff --check
commit
push
verify GitHub SHA
```

---

# 4. Superpowers workflow

Codex must discover and use installed equivalents of:

```text
brainstorming
writing-plans
test-driven-development
systematic-debugging
executing-plans
requesting-code-review
receiving-code-review
verification-before-completion
```

## Agent A — implementer

Agent A must:

- write an implementation manifest mapping every plan section to code/tests/evidence;
- write failing tests before each component;
- implement the clean formal namespace;
- expose tensor, gradient, performance, and intervention contracts;
- never issue the training authorization pass.

## Agent B — independent adversarial reviewer

Agent B must:

- start from this contract rather than Agent A's summary;
- inspect the actual diff and formal import graph;
- run the audit skill independently;
- deliberately search for legacy imports, orphan config, placeholder reports, false BF16 claims, wrong loss signs, dead gradients, target leakage, fake interventions, and evaluator mismatch;
- reject until every blocker has a regression test and dynamic proof.

Formal loop:

```text
plan draft
→ Agent B plan review
→ implementation with TDD
→ targeted tests
→ real direct-image smoke
→ mechanism/throughput/intervention gates
→ Agent B audit
→ fixes and regression tests
→ commit/push
→ full audit on exact clean pushed HEAD
→ review pass
→ foreground formal run
```

---

# 5. Clean formal namespace

Add:

```text
fate_x/acpr_dynflow_swin/
```

Formal entrypoints:

```text
python -m fate_x.engine.train_acpr_dynflow_swin
python -m fate_x.engine.eval_acpr_dynflow_swin
python -m fate_x.engine.run_acpr_dynflow_swin_preflight
python -m fate_x.engine.audit_acpr_dynflow_swin
python -m fate_x.engine.supervise_acpr_dynflow_swin_foreground
```

The formal import graph must not import or instantiate:

```text
fate_x.acpr_dynflow.model.ACPRDynFlowModel
fate_x.acpr_flow.model.ACPRFlowModel
fate_x.acpr_flow_v2.model.ACPRFlowCalV2Model
FlowTracePMTModel
TokenPMTAdapter
LogSinkhornTransport
legacy FlowCal/DynFlow trainers or losses
```

Legacy packages remain historical only.

---

# 6. Data and ADAPT-compatible metric contract

## 6.1 Data

```text
train:
  datasets_part/BDDX/training_32frames.yaml

test caption:
  datasets/BDDX/testing_32frames.yaml

test control:
  datasets_part/BDDX/testing_32frames.yaml
```

Verify caption/control sample-ID alignment.

## 6.2 Text tensorization

Use the ADAPT-compatible contract:

```text
frames = 32
resolution = 224
use_sep_cap = true
action max tokens = 15
explanation max tokens = 15
max sequence length = 30
mask probability = 0.5
max masked tokens = 45
WordPiece tokenizer from local bert-base-uncased
attention-mask semantics identical to ADAPT
```

Preserve raw action and justification strings in every batch.

## 6.3 Signal contract

Add one `BDDSignalCodec` used by:

```text
dataset conversion
training normalization
motion output decode
loss computation
official control evaluation
traffic intervention normalization
visualization
```

The codec must discover and write:

```text
signal order
valid/invalid marker
raw units/range
train mean/std
time length
official ADAPT evaluation behavior
```

Do not replace the official course metric with a circular metric unless the data-contract audit proves the original evaluator uses circular error. Optional circular diagnostics must be separate.

## 6.4 Baseline metric parity

Before formal training, run the strongest ADAPT reproduction checkpoint through the exact same test sample IDs and evaluators.

The new metric bridge must reproduce its existing evaluation JSON.

This comparison process is external to ACPR-DynFlow-Swin training; no ADAPT checkpoint or predictions enter the formal model.

## 6.5 Required primary metrics

Description/action and explanation separately:

```text
BLEU-1/2/3/4
METEOR
ROUGE-L
CIDEr
SPICE
```

Speed/course separately:

```text
RMSE
MAE
Acc@0.1
Acc@0.5
Acc@1
Acc@5
Acc@10
```

The heuristic action-text proxy is diagnostic only.

---

# 7. Typed contracts

Create `fate_x/acpr_dynflow_swin/types.py`.

## 7.1 Batch

```python
@dataclass
class DynFlowSwinBatch:
    frames: Tensor                     # [B,32,3,224,224]
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    masked_pos: Tensor
    masked_ids: Tensor
    control_target: Tensor | None      # [B,32,2]
    sample_ids: list[str]
    raw_actions: list[str]
    raw_justifications: list[str]
```

## 7.2 Backbone

```python
@dataclass
class SwinBackboneOutput:
    predicate_grid: Tensor             # [B,Tm,Hm,Wm,Dm], native stage
    final_grid: Tensor                 # [B,Th,7,7,Dh], native final stage
    temporal_global: Tensor            # [B,Th,Dh], spatially pooled
    dense_final_tokens: Tensor         # [B,Th*49,Dh], audit only
    forward_count: int
```

## 7.3 Predicate field

```python
@dataclass
class DynamicPredicateField:
    names: tuple[str, ...]              # exact 32
    query_states: Tensor                # [B,Tm,32,D]
    logits: Tensor                      # [B,Tm,32]
    probabilities: Tensor               # [B,Tm,32]
    tokens: Tensor                      # [B,Tm,32,D]
    evidence_maps: Tensor               # [B,Tm,32,Hm,Wm]
    confidence: Tensor                  # [B,Tm,32]
    centroid: Tensor                    # [B,Tm,32,2]
    relative_motion: Tensor             # [B,Tm-1,32,2]
    corridor_mass: Tensor               # [B,Tm,32,3]
    transfer_gate: Tensor               # [32]
```

## 7.4 Semantic consolidation

```python
@dataclass
class SemanticTokenConsolidation:
    slot_names: tuple[str, ...]         # global, longitudinal, left, right, residual
    assignment: Tensor                  # [B,Th,49,5], sum over slots = 1
    token_mass: Tensor                  # [B,Th,5]
    tokens: Tensor                      # [B,Th,5,Dtext]
    source_provenance: Tensor           # same assignment or compressed representation
    conservation_error: Tensor
```

Required exact property before projection:

\[
\sum_j mass_j z_j = \sum_i F_i
\]

within tolerance.

## 7.5 Traffic state

```python
@dataclass
class TrafficStateOutput:
    factor_names: tuple[str, ...]       # exact 13
    factor_tokens_native: Tensor        # [B,Tm,13,D]
    factor_logits: Tensor               # [B,Tm,13]
    factor_probs: Tensor                # [B,Tm,13]
    lateral_bias: Tensor                # [B,Tm,1]
    pattern_logits: Tensor              # [B,Tm,4]
    pattern_probs: Tensor               # [B,Tm,4]
    factor_to_predicate: Tensor         # [B,Tm,13,32]
    factor_to_corridor: Tensor          # [B,Tm,13,3]
    evidence_maps: Tensor               # [B,Tm,13,Hm,Wm]
    lag_weights: Tensor                 # [B,32,13,4]
    lag_aligned_tokens: Tensor          # [B,32,13,D]
    lineage: list[dict]
```

## 7.6 Motion output

```python
@dataclass
class MotionTransformerOutput:
    query_hidden: Tensor                # [B,32,768]
    global_prediction_normalized: Tensor # [B,32,2]
    source_attention: Tensor | None
```

## 7.7 Decision ledger

```python
@dataclass
class ExactDecisionLedger:
    signal_names: tuple[str, ...]
    global_prediction_normalized: Tensor
    raw_factor_contributions_normalized: Tensor  # [B,32,13,2]
    benefit_gate: Tensor                        # [B,32,2]
    gated_factor_contributions_normalized: Tensor
    final_prediction_normalized: Tensor
    global_prediction_raw: Tensor
    gated_factor_contributions_raw: Tensor
    final_prediction_raw: Tensor
    speed_factor_attention: Tensor
    course_factor_attention: Tensor
    benefit_target: Tensor | None
```

## 7.8 Text output

```python
@dataclass
class DynFlowSwinTextOutput:
    total_mlm_loss: Tensor
    action_loss: Tensor
    explanation_loss: Tensor
    action_logits: Tensor
    explanation_logits: Tensor
    action_to_factor_attention: Tensor
    explanation_to_factor_attention: Tensor
    generated_action: list[str] | None
    generated_explanation: list[str] | None
```

## 7.9 Formal output

```python
@dataclass
class ACPRDynFlowSwinOutput:
    total_loss: Tensor
    loss_components: dict[str, Tensor]
    backbone: SwinBackboneOutput
    predicates: DynamicPredicateField
    semantic_tokens: SemanticTokenConsolidation
    traffic: TrafficStateOutput
    motion: MotionTransformerOutput
    ledger: ExactDecisionLedger
    text: DynFlowSwinTextOutput
    diagnostics: dict[str, Any]
```

No positional tuple-tail parsing.

---

# 8. Full-capacity Video Swin-B

File:

```text
fate_x/acpr_dynflow_swin/video_swin_backbone.py
```

Use the repository's ADAPT-proven Video Swin implementation through `src/modeling/load_swin.py`, not `torchvision.swin3d_b`.

Initialization:

```text
Kinetics-600 only
```

Requirements:

- 32×224×224 input;
- one forward;
- no `.float()` inside the backbone;
- BF16 autocast respected;
- return a true native middle stage and true native final stage;
- no interpolation of final features masquerading as an intermediate stage;
- all Video Swin parameters trainable, as in ADAPT;
- backbone LR is much lower than head LR;
- optional activation checkpointing only if throughput probe shows it is needed to fit a stable batch.

Modify shared Swin code backward-compatibly to support:

```python
final, stages = swin(images, return_stages=(2, 3))
```

without changing default ADAPT output.

---

# 9. OIA predicate transfer

Files:

```text
predicate_ontology.py
predicate_transfer.py
configs/acpr_dynflow_swin_predicates.yaml
```

## 9.1 Exact ontology

Use the exact 32 names/order from `FATE-OIA/acpr_calalign_v1_2`.

## 9.2 Query construction

\[
q_k^0 =
q_k^{name}
+
g_k^{transfer} W_{OIA} q_k^{OIA}
+
r_k
\]

where:

- \(q_k^{name}\): true BERT encoding of the predicate name, projected to predicate dimension;
- \(q_k^{OIA}\): loaded learned OIA predicate query;
- \(W_{OIA}\): trainable explicit dimension mapper;
- \(g_k^{transfer}\): per-predicate sigmoid reliability gate initialized to 0.25;
- \(r_k\): trainable BDD-X residual.

This gate protects against an incompatible DINO-to-Swin query transfer while retaining static-to-dynamic knowledge transfer.

Write source checkpoint path, SHA256, tensor key, source shape, mapped shape, and loaded predicate order.

Review fails if the OIA checkpoint or ontology is unresolved.

---

# 10. Dynamic predicate field

File:

```text
dynamic_predicate_field.py
```

At native middle-stage time \(t\):

\[
q_{t,k}=q_k^0+GRU(e_{t-1,k},q_{t-1,k})
\]

\[
A_{t,k}=
entmax_{1.5}
\left(
q_{t,k}^{T}K_t/\sqrt d+b_k^{region}
\right)
\]

\[
e_{t,k}=\sum_{x,y} A_{t,k}(x,y)V_t(x,y)
\]

Use one shared GRU cell plus predicate embeddings.

## 10.1 Region priors

Port the OIA region-prior semantics as soft logit biases.

## 10.2 Ego-motion compensation

Use a bounded local correlation search on coarse native features to estimate one common image-plane shift per adjacent time step.

Subtract this shift from evidence-centroid displacement.

Do not use global Sinkhorn or full patch tracking.

## 10.3 Confidence

Combine:

```text
calibrated presence
entmax concentration
query temporal consistency
feature agreement
border/visibility
```

Confidence must be nonconstant and auditable.

---

# 11. Calibrated nnPU

File:

```text
nnpu_calalign.py
```

## 11.1 Complete rule schema

`configs/acpr_dynflow_swin_text_rules.yaml` must contain all 32 predicates.

Each entry has:

```text
supervision_mode:
  text_nnpu | structural_weak | consistency_only

positive_phrases
contradiction_phrases
exclusion_phrases
priority
```

Do not fabricate text positives for predicates that are not naturally mentioned. Use structural/consistency supervision for those.

## 11.2 Labels

From raw action and justification:

```text
explicit support       → positive
explicit contradiction → reliable negative
otherwise              → unlabeled
```

Known exclusion cases such as `light traffic` vs `traffic_light` require tests.

## 11.3 Nonnegative PU risk

\[
R_k^{nnPU}
=
\pi_k R_k^+
+
\max(0,R_k^{u-}-\pi_kR_k^{+-})
\]

No ordinary negative BCE on unlabeled entries.

## 11.4 Dynamic CalAlign

During train batches only, accumulate detached score histograms for known positives and reliable negatives.

At epoch end update per predicate:

```text
class prior
temperature
positive threshold
negative threshold
known-label precision/recall estimates
```

Soft probabilities enter downstream computation. Thresholds are used for intermediate output, confidence, pseudo-positive acceptance, and visualization.

Test evaluation cannot update any calibration state.

---

# 12. Predicate-guided mass-preserving semantic token consolidation

File:

```text
semantic_token_consolidator.py
```

Input native final grid:

```text
[B, Th, 49, Dh]
```

Create exactly five slots per native time:

```text
global_action
longitudinal
left_lateral
right_lateral
residual_context
```

Slot logits depend on:

```text
final token feature
downsampled predicate evidence
predicate probabilities/confidence
traffic/corridor priors available at that point
learned slot query
```

For each original token \(i\), normalize across slots:

\[
w_{i,j}=softmax_j(\ell_{i,j})
\]

so:

\[
\sum_jw_{i,j}=1
\]

Mass and token:

\[
m_j=\sum_iw_{i,j}
\]

\[
z_j=\frac{\sum_iw_{i,j}F_i}{m_j+\epsilon}
\]

Thus:

\[
\sum_jm_jz_j=\sum_iF_i
\]

before projection.

At 16 native time points this yields:

```text
16 × 5 = 80 semantic tokens
```

Save full provenance and mass.

The residual slot is mandatory and prevents the four semantic slots from discarding unmodeled context.

Do not prune/merge tokens inside Video Swin windows in V1.

---

# 13. Unified Pattern–Lag Traffic Reasoner

Files:

```text
predicate_covariates.py
mesoscopic_corridor_flow.py
pattern_lag_traffic_reasoner.py
```

This replaces disconnected router/reasoner/lag modules.

## 13.1 Predicate covariates

For every predicate/time:

```text
probability
first difference
second difference
centroid xy
ego-compensated centroid motion
left/center/right evidence mass
confidence
```

Use a shared homogenizer plus predicate semantic embedding.

## 13.2 Corridor flow

Using participant predicates only, compute distinct left/center/right:

```text
occupancy proxy
relative mobility
motion coherence
stopped tendency
queue pressure
gap openness
```

Signal lights and static road-region predicates can condition states but must not be counted as vehicles.

Do not replicate one global motion value across corridors.

## 13.3 Multi-scale temporal operator

Use depthwise causal temporal convolutions with dilation:

```text
1
2
4
```

and pointwise gated MLPs.

Fuse progressively from coarse context to fine sequence.

## 13.4 Operational pattern semantics

Patterns:

```text
stable
forming
releasing
oscillating
```

Weak detached targets come from smoothed first/second differences:

- positive trend → forming;
- negative trend → releasing;
- low first/second difference → stable;
- alternating/high second difference → oscillating.

Pattern output must feed factor construction and response lag.

## 13.5 Thirteen factors

Regime logits use softmax over four.

Phase logits use softmax over four.

Source logits use sigmoid over five.

Each factor token sparsely reads predicate and corridor tokens with entmax and has an evidence map and lineage.

A rank-4 compositional interaction forms a joint traffic context without defining a large flat combination class.

## 13.6 Response lag

For decision step \(t\), factor \(k\), lags \(0..3\):

\[
\lambda_{t,k,\ell}
=
softmax_{\ell}(h_t^T W_{\ell} f_{t-\ell,k})
\]

Out-of-range lags are masked.

The lag-aligned factor tokens are used by the decision ledger and text explanation.

---

# 14. ADAPT-level independent query motion transformer

File:

```text
query_motion_transformer.py
```

Do not use the legacy GRU control head and do not load ADAPT sensor weights.

Use BERT-base hidden size, heads, feed-forward width, and 12 encoder layers, independently initialized.

Input sequence:

```text
32 learned output-time query tokens
16 temporally pooled Video Swin final tokens
80 mass-preserving semantic tokens
```

Use token-type and temporal-position embeddings.

The first 32 output hidden states correspond to output-time queries and decode the independent global speed/course trajectory.

This retains ADAPT-level motion-transformer capacity while reducing its visual source sequence from hundreds of redundant grid tokens to semantic, mass-preserving tokens.

Control targets never enter the forward predictor.

---

# 15. Benefit-constrained exact decision ledger

File:

```text
decision_ledger.py
```

## 15.1 Separate readers

Speed and course use separate query/key/value and contribution heads.

Speed receives a soft prior toward:

```text
queue/following/signal/lead vehicle/longitudinal corridor/forming/releasing
```

Course receives a soft prior toward:

```text
left/right gaps/side vehicles/merge/turn/lateral bias/corridor advantage
```

Priors are logit biases, not hard masks.

## 15.2 Raw factor contribution

\[
\Delta U^{raw}_{t,k}=a_{t,k}d_{t,k}
\]

## 15.3 Benefit gate

One gate per signal/time:

\[
\alpha_t=\sigma(g(h_t^{global},f_t))
\]

The gate scales all factor contributions for that signal:

\[
\Delta U^{gated}_{t,k}=\alpha_t\Delta U^{raw}_{t,k}
\]

## 15.4 Final output

\[
U_t^{final}
=
U_t^{global}
+
\sum_k\Delta U^{gated}_{t,k}
\]

## 15.5 Complementarity target

\[
L_{residual}
=
Huber
\left(
\sum_k\Delta U^{raw}_{t,k},
stopgrad(Y_t-U_t^{global})
\right)
\]

## 15.6 Benefit target

Compute detached candidate improvement:

\[
b_t =
\ell(U_t^{global},Y_t)
-
\ell(U_t^{global}+\sum_k\Delta U^{raw}_{t,k},Y_t)
\]

\[
\alpha_t^*=\sigma(b_t/\tau)
\]

Use BCE/Huber between gate and detached benefit target.

## 15.7 Non-degradation hinge

\[
L_{safe}
=
confidence_{flow}
\cdot
ReLU(
\ell(U^{final},Y)
-
stopgrad(\ell(U^{global},Y))
+
margin
)
\]

This trains the flow branch to help or abstain.

Do not use contribution magnitude itself as a residual loss.

---

# 16. ADAPT-compatible autoregressive action/explanation decoder

Files:

```text
text_decoder.py
contribution_reason_adapter.py
```

Use the repository's `BertForImageCaptioning`/sep-cap generation path and generic BERT-base initialization.

Do not load ADAPT task weights.

## 16.1 Text visual sequence

Supply:

```text
80 semantic consolidated tokens
13 pooled traffic-factor tokens
13 contribution-summary tokens
```

Dynamically resize the multimodal attention mask.

## 16.2 Contribution-reason adapter hook

Add a backward-compatible generic hook after multimodal BERT hidden states and before the LM prediction head.

Action text tokens read:

```text
global motion summary
traffic factors
```

Explanation text tokens read:

```text
traffic factors
supporting predicates
exact signed decision contributions
generated action hidden
```

Image hidden states remain unchanged by the adapter.

## 16.3 Gradient boundary

Explanation receives detached global motion context:

```python
global_context_for_explanation = global_motion_hidden.detach()
```

Explanation gradients still update predicate, traffic, contribution adapter, and text decoder.

## 16.4 Separate losses

Compute masked-token loss separately for action and explanation masks.

One duplicated text loss assigned to both fields is forbidden.

## 16.5 Contribution alignment

Create per-factor target:

\[
q_k=
Normalize
\sum_t
\left(
|\Delta speed_{t,k}|/\sigma_s
+
|\Delta course_{t,k}|/\sigma_c
\right)
\]

Align explanation-to-factor attention with Jensen–Shannon divergence.

## 16.6 Generation

Inference:

```text
autoregressively generate action
then autoregressively generate explanation conditioned on generated action state
```

No GT action/justification at inference.

Use the same tokenizer, max lengths, beam setting, and evaluator as ADAPT.

---

# 17. Losses

File:

```text
fate_x/losses/acpr_dynflow_swin_losses.py
```

Signal-specific losses are calculated by signal name, never by assumed index.

Recommended weights:

```text
final_speed_normalized       1.00
final_course_normalized      1.00
global_speed_normalized      0.50
global_course_normalized     0.50
flow_residual_speed          0.25
flow_residual_course         0.25
benefit_gate                 0.05
non_degradation              0.10
control_delta_match          0.10

action_text                  0.50
explanation_text             0.50

predicate_nnpu               0.08
predicate_transfer_anchor    0.02
pattern_semantic             0.04
traffic_state_semantic       0.03
contribution_alignment       0.04
contribution_group_sparsity  0.002
contribution_smoothness      0.005
```

## 17.1 Control

Use train-standardized SmoothL1/Huber separately for speed and course.

First-difference loss:

\[
Huber(\Delta U^{final},\Delta Y)
\]

not `|ΔU|`.

## 17.2 State semantics

Use valid weak semantic targets and grammar contradictions, not mean-probability minimization.

## 17.3 Sparsity

Use very low group sparsity on factor groups only after residual supervision exists. It cannot be the main flow objective.

Explicitly absent:

```text
HardPair
SCST
PCGrad/Gradient Firewall
global Sinkhorn
second contextual BERT
ADAPT distillation
unknown-negative BCE
```

---

# 18. Trainer, optimizer, and precision

Files:

```text
fate_x/engine/train_acpr_dynflow_swin.py
fate_x/engine/acpr_dynflow_swin_data.py
```

## 18.1 Formal run

```text
epochs = 16
precision = native BF16 autocast
optimizer = AdamW
warm-up = 10%
scheduler = linear decay
gradient clip = 1.0
effective batch target = 64
metric early stop = false
```

## 18.2 Fixed trainability

All epochs:

```text
full Video Swin-B trainable
predicate/traffic/consolidation/motion/ledger trainable
BERT caption decoder trainable
generic BERT lower layers use lower LR, not stage freezing
```

## 18.3 Learning rates

```text
new traffic/ledger/motion heads       2.0e-4
predicate/query/temporal modules      1.0e-4
caption BERT and LM head              5.0e-5
Video Swin-B                          1.0e-5
```

Use explicit parameter groups. Every trainable parameter appears exactly once.

## 18.4 Precision runtime

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(batch)
```

No forced `.float()` in the backbone.

Enable:

```text
TF32 matmul where supported
cuDNN benchmark
non-blocking H2D
fused AdamW only if runtime supports it and parity test passes
```

Gradient checkpointing is off by default. The memory probe may enable Video Swin activation checkpointing only if no throughput-optimal batch fits under the hard cap.

## 18.5 Resume

Checkpoint contains:

```text
model
optimizer
scheduler
epoch
global/optimizer step
RNG states
nnPU/CalAlign state
signal codec
best records
config/Git hashes
```

Reject `.tmp`.

---

# 19. Data and throughput engineering

Formal training must run in WSL/Linux, not native Windows Python.

DataLoader:

```text
num_workers = 8
persistent_workers = true
prefetch_factor = 4
pin_memory = true
non_blocking CUDA transfer = true
```

Implement a CUDA prefetcher only if measured to improve throughput.

Do not read/decode the same frame sequence twice per batch.

---

# 20. Memory and throughput gate

The user's desired operating range is around 40 GiB, but selection is based on throughput, not maximal allocation.

Target:

```text
preferred peak reserved: 36–42 GiB
hard cap: 44 GiB
minimum headroom: 3 GiB
```

Probe:

```text
batch 8 / accumulation 8
batch 6 / accumulation 11
batch 4 / accumulation 16
batch 3 / accumulation 22
batch 2 / accumulation 32
```

For each candidate:

```text
10 warm-up steps
100 measured real forward/backward steps
direct 32-frame images
all formal losses
native BF16
```

Record:

```text
data_time
Video Swin time
predicate/traffic time
consolidation time
motion time
text time
backward time
optimizer time
samples/s
peak allocated/reserved
projected train hours/epoch
projected test-eval hours/epoch
```

Select the highest samples/s candidate satisfying memory and finite-step constraints.

Formal launch is blocked unless:

```text
projected train time per epoch <= 4 hours
data_time share <= 20%
no NaN/Inf
no skipped optimizer step
BF16 proven active
```

No dummy memory allocation.

---

# 21. Per-epoch evaluation and checkpoints

At every epoch end:

1. atomically save `checkpoint_latest.pth`;
2. run full test action/description generation;
3. run full test explanation generation;
4. run full test speed/course evaluation;
5. run lightweight 256-sample traffic influence audit;
6. save best checkpoints;
7. continue regardless of metric direction.

No validation loader.

Save:

```text
checkpoint_latest.pth
checkpoint_best_text.pth
checkpoint_best_control.pth
checkpoint_best_joint.pth
checkpoint_best_test.pth
```

## 21.1 Best text

Maximize:

```text
CIDEr_description + CIDEr_explanation
```

## 21.2 Best control

Minimize:

\[
0.5 RMSE_s/RMSE_s^{ADAPT}
+
0.5 RMSE_c/RMSE_c^{ADAPT}
\]

using the measured ADAPT reproduction.

## 21.3 Best test

Decision-first, text-safe lexicographic selection.

Eligible if:

```text
CIDEr_description + CIDEr_explanation
>= 0.85 × measured ADAPT CIDEr sum
```

Among eligible:

1. minimum normalized control score;
2. higher explanation CIDEr;
3. higher description CIDEr;
4. lower speed RMSE;
5. lower course RMSE.

If none is eligible, choose minimum normalized control score and record `text_floor_not_met=true`.

---

# 22. Traffic-flow decision influence verification

The additional evaluation does not replace ADAPT metrics.

## 22.1 Lightweight per-epoch audit

Fixed 256 test samples:

```text
full
global-only / all-flow-off
top-factor-off
temporal-reverse
lag-disabled
```

Not used for checkpoint selection.

## 22.2 Full best-checkpoint audit

On `checkpoint_best_test.pth`:

```text
global-only/all-flow-off
regime-off
phase-off
source-off
individual factor-off
predicate-off
evidence-tube-off
equal-mass random evidence deletion, 5 seeds
temporal shuffle
temporal reverse
lag disabled
last-frame-only
semantic-consolidation residual-only
```

Interventions rerun from the earliest affected layer.

## 22.3 Metrics

Traffic Flow Utilization:

\[
TFU_s=RMSE_s(flowoff)-RMSE_s(full)
\]

\[
TFU_c=RMSE_c(flowoff)-RMSE_c(full)
\]

Evidence Specificity:

\[
ES=\Delta Error_{evidence}-E[\Delta Error_{random}]
\]

Temporal Direction Reliance:

\[
TDR=Error(reverse/shuffle)-Error(full)
\]

Lag Necessity:

\[
LN=Error(lag0)-Error(learnedlag)
\]

Ledger Fidelity:

Spearman correlation between:

```text
absolute exact contribution
factor-off output effect
```

Text–Decision Consistency:

```text
explanation factor attention
vs normalized exact ledger contributions
```

Report conditional subsets:

```text
all
traffic signal
queue/following
turn left
turn right
merge/lane change
vulnerable road user
```

Claims are model-level dependence, not real-world causality.

---

# 23. Dynamic Traffic Decision Ledger visualization

Files:

```text
fate_x/explain/acpr_dynflow_swin_renderer.py
fate_x/explain/acpr_dynflow_swin_atlas.py
fate_x/explain/acpr_dynflow_swin_faithfulness.py
```

Each selected case produces PNG + source JSON.

Panels:

1. **Predicate Evidence Tubes**
   - eight keyframes;
   - stable predicate colors;
   - real named evidence maps.

2. **Semantic Token Consolidation**
   - original 16×7×7 tokens;
   - assignment to five semantic slots;
   - mass per slot;
   - residual coverage;
   - downstream factor/decision lineage.

3. **Corridor Flow Ribbons**
   - left/center/right occupancy proxy;
   - mobility;
   - stopped tendency;
   - queue pressure;
   - gap openness.

4. **Pattern/State Lattice**
   - stable/forming/releasing/oscillating;
   - regime/phase/source;
   - lateral bias;
   - confidence and support predicates.

5. **Event–Response Lag Ribbon**
   - traffic event;
   - selected lag distribution;
   - resulting control contribution.

6. **Exact Signed Decision Waterfall**
   - global speed/course;
   - each factor's gate-weighted contribution;
   - benefit gate;
   - exact final reconstruction.

7. **Contribution-Aligned Text**
   - action/explanation;
   - token-to-factor attention;
   - actual signed decision effect.

8. **Counterfactual Twin**
   - full;
   - flow-off;
   - factor-off;
   - evidence-off;
   - temporal reverse;
   - equal-mass random comparison.

Atlas:

```text
state → action matrix
factor → normalized speed effect
factor → normalized course effect
response-lag distributions
factor transition graph
explanation phrase → contribution factor
evidence vs random effect distributions
global-vs-flow decision coverage
representative successes and failures
```

No manual boxes, fabricated values, template-only counterfactuals, or record-dump-only atlas.

---

# 24. Files to add

```text
configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml
configs/acpr_dynflow_swin_predicates.yaml
configs/acpr_dynflow_swin_text_rules.yaml
configs/acpr_dynflow_swin_traffic_grammar.yaml

fate_x/acpr_dynflow_swin/__init__.py
fate_x/acpr_dynflow_swin/config.py
fate_x/acpr_dynflow_swin/types.py
fate_x/acpr_dynflow_swin/signal_codec.py
fate_x/acpr_dynflow_swin/video_swin_backbone.py
fate_x/acpr_dynflow_swin/predicate_ontology.py
fate_x/acpr_dynflow_swin/predicate_transfer.py
fate_x/acpr_dynflow_swin/ego_motion.py
fate_x/acpr_dynflow_swin/dynamic_predicate_field.py
fate_x/acpr_dynflow_swin/nnpu_calalign.py
fate_x/acpr_dynflow_swin/semantic_token_consolidator.py
fate_x/acpr_dynflow_swin/predicate_covariates.py
fate_x/acpr_dynflow_swin/mesoscopic_corridor_flow.py
fate_x/acpr_dynflow_swin/pattern_lag_traffic_reasoner.py
fate_x/acpr_dynflow_swin/query_motion_transformer.py
fate_x/acpr_dynflow_swin/decision_ledger.py
fate_x/acpr_dynflow_swin/contribution_reason_adapter.py
fate_x/acpr_dynflow_swin/text_decoder.py
fate_x/acpr_dynflow_swin/interventions.py
fate_x/acpr_dynflow_swin/model.py

fate_x/losses/acpr_dynflow_swin_losses.py

fate_x/engine/acpr_dynflow_swin_data.py
fate_x/engine/eval_adapt_reference_dynflow.py
fate_x/engine/train_acpr_dynflow_swin.py
fate_x/engine/eval_acpr_dynflow_swin.py
fate_x/engine/run_acpr_dynflow_swin_preflight.py
fate_x/engine/audit_acpr_dynflow_swin.py
fate_x/engine/probe_acpr_dynflow_swin_throughput.py
fate_x/engine/export_acpr_dynflow_swin_visuals.py
fate_x/engine/build_acpr_dynflow_swin_atlas.py
fate_x/engine/supervise_acpr_dynflow_swin_foreground.py

fate_x/explain/acpr_dynflow_swin_renderer.py
fate_x/explain/acpr_dynflow_swin_atlas.py
fate_x/explain/acpr_dynflow_swin_faithfulness.py

scripts/FATE_X_acpr_dynflow_swin_v1_foreground.ps1
scripts/FATE_X_acpr_dynflow_swin_v1_foreground.sh

docs/runbooks/ACPR_DynFlow_Swin_V1_Implementation_Plan.md
docs/runbooks/ACPR_DynFlow_Swin_V1_Implementation_Manifest.json
.codex/skills/acpr-dynflow-swin-implementation-audit/SKILL.md
```

---

# 25. Shared files that may be modified

Only as required and backward-compatibly:

```text
src/modeling/load_swin.py
src/modeling/video_swin/swin_transformer.py
src/layers/bert/modeling_bert.py
src/datasets/vision_language_tsv.py
src/datasets/vl_dataloader.py
src/datasets/caption_tensorizer.py
.gitignore
```

Do not expand the legacy `src/tasks/run_adapt.py` into the formal trainer.

---

# 26. Required tests

Create `tests/acpr_dynflow_swin/`.

## Isolation/config

```text
test_formal_import_graph.py
test_config_binding.py
test_no_legacy_dynflow_import.py
test_no_adapt_task_checkpoint_in_training.py
test_direct_image_no_cache.py
test_named_output_contract.py
```

## Data/evaluation

```text
test_adapt_text_contract.py
test_adapt_metric_parity.py
test_signal_codec.py
test_caption_control_sample_alignment.py
test_test_only_protocol.py
```

## Backbone/precision

```text
test_video_swin_native_stages.py
test_single_backbone_forward.py
test_backbone_bf16_no_forced_fp32.py
test_backbone_trainable_groups.py
```

## Predicates/PU

```text
test_exact_32_predicates.py
test_oia_query_transfer.py
test_transfer_reliability_gate.py
test_region_priors.py
test_recurrent_predicate_field.py
test_entmax_evidence.py
test_ego_motion_compensation.py
test_nnpu_rules_all_predicates.py
test_unknown_not_negative.py
test_nnpu_nonnegative_risk.py
test_calalign_train_only_resume.py
```

## Consolidation/traffic

```text
test_mass_preserving_consolidation.py
test_consolidation_provenance.py
test_residual_slot_preserves_context.py
test_corridor_flow_distinct.py
test_pattern_semantics.py
test_pattern_path_reaches_states.py
test_traffic_factor_composition.py
test_response_lag.py
test_state_evidence_lineage.py
```

## Decision/text

```text
test_query_motion_transformer.py
test_motion_target_independence.py
test_signal_specific_losses.py
test_exact_ledger_identity.py
test_benefit_gate_target.py
test_non_degradation_loss_direction.py
test_speed_course_factor_routing.py
test_adapt_compatible_autoregressive_decode.py
test_separate_action_explanation_losses.py
test_explanation_global_detach.py
test_contribution_alignment.py
test_inference_no_gt_text.py
```

## Training/performance

```text
test_optimizer_groups.py
test_bf16_runtime.py
test_scheduler_resume.py
test_atomic_checkpoint_resume.py
test_best_selectors.py
test_throughput_probe_is_real.py
```

## Intervention/visual/supervisor

```text
test_intervention_recompute.py
test_equal_mass_random.py
test_ledger_factor_off_consistency.py
test_visual_canvas_schema.py
test_atlas_schema.py
test_review_pass_binding.py
test_foreground_supervisor.py
test_e2e_direct_image_smoke.py
```

---

# 27. Blocking preflight gates

## Gate A — Git/import/config

- current worktree/branch only;
- clean;
- local SHA equals GitHub SHA;
- formal import graph excludes legacy paths;
- every formal config field has a runtime consumer;
- all external paths resolved.

## Gate B — ADAPT data/metric parity

- identical test IDs;
- identical tensorization;
- metric bridge reproduces existing ADAPT outputs;
- signal order and invalid mask match.

## Gate C — independence

- no ADAPT task weights/logits/features read;
- only Kinetics, generic BERT, and OIA queries loaded.

## Gate D — static/unit tests

```bash
python -m compileall -q fate_x src
python -m pytest tests/acpr_dynflow_swin -q
python -m pytest tests -q
git diff --check
```

## Gate E — real direct-image smoke

At least:

```text
8 real train samples
8 real test samples
8 optimizer steps
batch 1
beam 1
```

Require checkpoint and actual text/control evaluation.

## Gate F — gradient chain

Finite nonzero gradients for every intended trainable module.

No gradients for frozen calibration accumulators or text-rule targets.

## Gate G — 128-sample mechanism fit

Bounded updates on deterministic 128 train samples.

Require decreasing:

```text
global speed/course
final speed/course
residual target
action
explanation
known-label nnPU
pattern/state
contribution alignment
```

Reject collapse:

```text
all predicates identical
all patterns stable
all factors constant
all flow contributions zero
all benefit gates zero
course ignores lateral factors
explanation attention unrelated to contribution
```

## Gate H — mass/ledger identities

Require exact semantic mass conservation and exact decision-ledger reconstruction.

## Gate I — temporal/lag

Reverse/shuffle affects phase and output. Known delayed synthetic event recovers lag.

## Gate J — real interventions

Flow/factor/evidence interventions alter actual outputs; evidence deletion exceeds equal-mass random on an intended branch in the smoke set.

## Gate K — visualization

One real complete Canvas and mini Atlas.

## Gate L — throughput/memory

100 real measured steps; projected epoch <=4 hours; target memory/finite constraints.

## Gate M — independent audit

Only Agent B may issue the review pass.

---

# 28. Review pass

Required:

```text
.background_runs/acpr_dynflow_swin_v1_preflight/
REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt
```

It binds:

```text
worktree
branch
clean local SHA
GitHub SHA
plan/config/skill/manifest hashes
all gate reports
selected batch/accumulation
projected epoch time
```

Any code/config/test/script change invalidates it.

---

# 29. Foreground supervisor

Sequence:

```text
verify review pass and exact SHA
evaluate ADAPT reference
reconfirm throughput/memory selection
run 16-epoch training
full test after every epoch
maintain best/latest checkpoints
run full traffic influence audit on best-test
render Canvas and Atlas
update canonical records
verify GitHub SHA
write run_complete.json
```

Forbidden:

```text
Start-Process
Start-Job
schtasks
nohup
shell &
DETACHED_PROCESS
hidden window
metric-based early stop
```

Required:

- attached child;
- live stdout/stderr;
- heartbeat every 60 seconds;
- no stop for bad metrics;
- transient I/O/evaluator retry up to three times;
- OOM fallback to next audited candidate;
- atomic latest resume;
- `.tmp` rejection;
- user stop sentinel:
  `<run>/control/STOP_REQUESTED_BY_USER`.

Codex must not create the stop sentinel without an explicit user command.

For a reproducible code defect:

```text
preserve checkpoint/log
systematic debugging
add regression test
fix
commit/push
invalidate review pass
rerun full audit
resume latest
continue foreground
```

Normal completion only after the entire configured suite, explicit user stop, or unrecoverable hardware/OS failure.

---

# 30. Formal launch

After review pass:

```powershell
Set-Location E:\sbw\FATE_Drive\fate_x_acpr_dynflow_v1_worktree

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_X_acpr_dynflow_swin_v1_foreground.ps1 `
  -Config configs\acpr_dynflow_swin_v1_bddx_32f_224.yaml `
  -RequireReviewPass
```

The process remains attached.

---

# 31. Required epoch artifacts

```text
epoch_XXX/
  description_metrics.json
  explanation_metrics.json
  control_metrics.json
  loss_components.jsonl
  gradient_norms.json
  predicate_stats.json
  predicate_calibration.json
  semantic_token_stats.json
  traffic_state_stats.json
  response_lag_stats.json
  decision_ledger_stats.json
  lightweight_flow_influence.json
  predictions_description.tsv
  predictions_explanation.tsv
  control_predictions.pt
  fixed_case_intermediate_outputs.jsonl
```

Run root:

```text
config_resolved.yaml
run_manifest.json
git_provenance.json
adapt_reference_metrics.json
adapt_metric_parity.json
signal_contract.json
oia_transfer_report.json
optimizer_groups.json
throughput_memory_probe.json
checkpoint_latest.pth
checkpoint_best_text.pth
checkpoint_best_control.pth
checkpoint_best_joint.pth
checkpoint_best_test.pth
train.log
supervisor_live_status.json
supervisor_decisions.jsonl
run_complete.json
```

Missing values require a reason; no fake zeros.

---

# 32. Definition of implementation complete

Implementation is complete only when:

- the clean formal namespace exists;
- formal import graph excludes all legacy models;
- metric parity is proven;
- one real Video Swin-B forward provides true native stages;
- native BF16 is proven;
- all 32 OIA predicates and query transfer execute;
- nnPU receives real positive/reliable-negative examples;
- semantic consolidation is mass preserving;
- pattern output changes traffic states;
- response lag changes decisions;
- motion transformer is target independent and ADAPT-capacity;
- exact ledger identity holds;
- benefit gate and safe loss have correct direction;
- text generation is autoregressive and ADAPT-compatible;
- action/explanation losses are separate;
- explanation is aligned with real contributions;
- interventions rerun real downstream paths;
- Canvas/Atlas use real tensors;
- all tests/mechanism/performance gates pass;
- review pass binds the exact clean pushed SHA.

File existence and placeholder JSON are insufficient.

---

# 33. Definition of experiment complete

The formal experiment is complete only after:

- ADAPT reference evaluation;
- all 16 epochs;
- full test after every epoch;
- all best/latest checkpoints;
- best-test full traffic-flow audit;
- full Dynamic Traffic Decision Ledger export;
- dataset Atlas;
- canonical record updates;
- local/GitHub SHA equality;
- `run_complete.json`.

---

# 34. Hard audit failures

Training remains blocked if any is true:

```text
legacy acpr_dynflow model imported
ADAPT task checkpoint loaded by formal model
torchvision Swin3D used as formal backbone
backbone forces FP32
intermediate feature is interpolated final feature
more than one Video Swin forward per batch
predicate names anonymous
OIA checkpoint unresolved
PU masks hard-coded all-unlabeled
unknown treated as negative
pattern output disconnected from traffic states
left/center/right traffic values replicated
lag configured but unused
token consolidation not mass preserving
dense 784-token text/motion head used despite consolidation config
motion target enters forward
speed/course share one duplicated loss
factor magnitude used as residual loss
prediction variation minimized without target variation
ledger does not exactly reconstruct final output
benefit gate has no real target
action/explanation share one duplicated text loss
text generation is position-wise argmax
explanation gradient reaches global motion stream
attention presented as exact contribution
intervention changes only display tensors
metric parity missing
validation loader created
test updates model/calibration
BF16 configured but not active
memory/throughput report is fabricated
projected epoch exceeds 4 hours
renderer/atlas placeholder
review report auto-passes missing evidence
supervisor detaches or metric-stops
worktree dirty
local/GitHub SHA mismatch
review pass from another SHA
```
