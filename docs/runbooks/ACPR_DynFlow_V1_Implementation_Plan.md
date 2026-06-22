# ACPR-DynFlow V1
## Codex code-level implementation, review, experiment, and foreground-supervision contract

**Repository:** `https://github.com/d2116056543-lab/FATE-X`  
**Source worktree:** `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree`  
**Source branch:** `flowtrace_pmt_v1`  
**New worktree:** `E:\sbw\FATE_Drive\fate_x_acpr_dynflow_v1_worktree`  
**New Git branch:** `acpr_dynflow_v1`  
**Formal method:** `ACPR-DynFlow V1`  
**Formal config:** `configs/acpr_dynflow_v1_bddx_32f_224.yaml`

The new worktree must be created from the exact current HEAD of `flowtrace_pmt_v1` after the source worktree is clean and synchronized with GitHub. Do not use `main` as the base and do not modify the source worktree after creation.

---

# 0. Formal research objective

Implement an independent BDD-X model that extends the user's successful BDD-OIA ACPR idea from static images to video:

```text
static named predicates
→ dynamic named predicate trajectories
→ ego-centric mesoscopic traffic-flow states
→ factor-decomposed vehicle decisions
→ contribution-aligned action/explanation generation
```

The model is not an ADAPT residual plug-in. ADAPT is used only as:

1. an external comparison baseline;
2. the source of the official BDD-X data/evaluation protocol;
3. a source of reusable generic implementation components such as Video Swin and BERT captioning code.

The formal ACPR-DynFlow training path must not load an ADAPT task checkpoint, consume ADAPT predictions, or predict a residual relative to ADAPT.

Allowed generic initialization:

```text
Video Swin Kinetics-600 pretraining
BERT-base initialization
the user's BDD-OIA ACPR-CalAlign predicate-query checkpoint
```

Forbidden formal initialization:

```text
ADAPT model.bin
FlowCal V1/V2 checkpoints
FlowTrace PMT checkpoints
cached ADAPT predictions or logits
```

The complete formal path is:

\[
V_{1:T}
\rightarrow
P_{1:T}
\rightarrow
C_{1:T}
\rightarrow
F_{1:T}
\rightarrow
D_{1:T}
\rightarrow
\{U,A,E\}
\]

where:

- \(V\): 32-frame direct-image input;
- \(P\): 32 named dynamic ACPR predicates;
- \(C\): homogenized predicate covariate sequences;
- \(F\): compositional traffic regime/phase/source states with response lag;
- \(D\): exact signed state contributions to speed/course;
- \(U\): continuous speed/course trajectories;
- \(A\): original BDD-X action/description;
- \(E\): original BDD-X justification/explanation.

---

# 1. Non-negotiable invariants

## 1.1 Model independence

Formal ACPR-DynFlow training must:

- instantiate its own model;
- load only generic Kinetics/BERT pretraining and the user's OIA predicate prior;
- train its own decision head;
- train its own text head;
- never call ADAPT forward during training;
- never use ADAPT outputs as targets, residual bases, or inputs.

A separate baseline-evaluation command may instantiate ADAPT.

## 1.2 ACPR continuity

The formal model must contain:

- the exact 32 named predicates from `FATE-OIA/acpr_calalign_v1_2`;
- transferred OIA predicate queries/prototypes;
- region priors;
- entmax evidence maps;
- positive–unlabeled predicate supervision;
- predicate-specific calibration thresholds;
- a predicate/reason/decision chain.

Anonymous `predicate_00` slots are forbidden.

## 1.3 Decision-first objective

The primary research objective is vehicle decision prediction.

The model must report the exact ADAPT-compatible text and continuous-control metrics, but checkpoint selection gives priority to control subject to a text-performance floor.

## 1.4 Exact decision decomposition

The final speed/course values must be exactly reconstructible as:

\[
U_t =
U_t^{global}
+
\sum_k \Delta U_{t,k}^{flow}
\]

The saved state contributions must be the tensors used in the real forward computation, not a post-hoc attribution approximation.

## 1.5 Direct-image execution

Formal training loads direct image tensors:

```text
[B, 32, 3, 224, 224]
```

No learned visual-feature cache, token cache, ADAPT-logit cache, or model-output cache is permitted.

Small train-only text/calibration statistics are allowed:

```text
predicate class priors
threshold histograms
control mean/std
OIA query-transfer metadata
```

## 1.6 Single unified training run

There is no semantic stage, motion stage, joint stage, or SCST stage.

All trainable ACPR-DynFlow components are active in one fixed architecture for all epochs. Warm-up of learning rate or bounded contribution scales is allowed; changing the model or trainable module set by epoch is not.

## 1.7 Formal evaluation protocol requested by the user

- evaluate `test` after every epoch;
- do not instantiate a validation loader in the formal run;
- select best checkpoints using test;
- record `protocol_tag=test_selected_user_requested`;
- do not stop because metrics decline or plateau.

## 1.8 Foreground-only execution

The formal supervisor and trainer remain attached to the foreground terminal. No detached/background execution.

---

# 2. Required context and record policy

Before any Git command, code edit, test, evaluation, training, or push, Codex must read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md

<source worktree>\docs\acpr_flowcalpp\ACPR_FlowCalPP_task_plan.md
<source worktree>\docs\acpr_flowcalpp\ACPR_FlowCalPP_findings.md
<source worktree>\docs\acpr_flowcalpp\ACPR_FlowCalPP_progress.md
```

After creating the new worktree, continue updating the existing canonical ACPR task/findings/progress ledgers. Do not create new experiment-status Markdown files.

The following are allowed because they are implementation runbooks, not status ledgers:

```text
docs/runbooks/ACPR_DynFlow_V1_Implementation_Plan.md
docs/runbooks/ACPR_DynFlow_V1_Implementation_Manifest.json
.codex/skills/acpr-dynflow-implementation-audit/SKILL.md
```

Never commit:

```text
.background_runs/
active run directories
datasets
checkpoints
*.pt
*.pth
prediction TSV/JSONL
feature/token caches
generated images/videos
memory probes
```

---

# 3. Worktree and branch creation

Use the Superpowers Git-worktree skill.

## 3.1 Source safety inspection

```powershell
$Root = "E:\sbw\FATE_Drive"
$Source = "$Root\fate_x_flowtrace_pmt_v1_worktree"
$Target = "$Root\fate_x_acpr_dynflow_v1_worktree"
$SourceBranch = "flowtrace_pmt_v1"
$TargetBranch = "acpr_dynflow_v1"

Set-Location $Source
git branch --show-current
git status --short
git remote -v
git fetch github
git rev-parse HEAD
git ls-remote github "refs/heads/$SourceBranch"
```

Required:

```text
branch == flowtrace_pmt_v1
source local HEAD == github/flowtrace_pmt_v1
```

If the source is dirty:

1. inspect every change;
2. save `git diff` under ignored `.background_runs/pre_dynflow_snapshot/`;
3. separate source changes from run artifacts;
4. test intended source changes;
5. commit and push intended source changes on `flowtrace_pmt_v1`;
6. verify source local/remote SHA equality.

Forbidden:

```text
git reset --hard
git clean -fd
discarding unknown edits
copying an uncommitted worktree as if it were reproducible
```

## 3.2 Create the new worktree from the current source HEAD

```powershell
$BaseSha = (git -C $Source rev-parse HEAD).Trim()

if (Test-Path $Target) {
    throw "Target worktree already exists; inspect it instead of overwriting it."
}

git -C $Source worktree add -b $TargetBranch $Target $BaseSha
git -C $Target push -u github "$TargetBranch`:$TargetBranch"

git -C $Target rev-parse HEAD
git -C $Target status --short
git -C $Target ls-remote github "refs/heads/$TargetBranch"
```

Write the source branch, source SHA, new branch, new SHA, and creation time to:

```text
.background_runs/acpr_dynflow_v1_preflight/worktree_provenance.json
```

The source worktree remains read-only for the rest of this task.

---

# 4. Required Superpowers workflow

Codex must discover and use the installed equivalents of:

```text
using-git-worktrees
brainstorming
writing-plans
test-driven-development
systematic-debugging
executing-plans
requesting-code-review
receiving-code-review
verification-before-completion
```

Use two roles.

## Agent A — implementer

Agent A must:

- inspect the real current branch import graph;
- write the implementation manifest;
- write failing tests before each component;
- implement only in the new worktree;
- expose tensor/gradient/intervention contracts;
- never authorize or start formal training.

## Agent B — independent adversarial reviewer

Agent B must:

- start from this plan and the audit skill;
- inspect the actual diff and import graph;
- run all static and dynamic gates;
- reject dead code, orphan config, placeholders, target leakage, false metric parity, and fake interventions;
- issue the review pass only for the exact clean pushed SHA.

Required loop:

```text
Agent A plan
→ Agent B plan review
→ Agent A implementation
→ component tests
→ direct-image integration smoke
→ mechanism tests
→ Agent B audit
→ blocker fixes with regression tests
→ commit/push
→ Agent B reruns audit on exact pushed HEAD
→ review pass
→ foreground formal experiment
```

---

# 5. Formal namespace and legacy isolation

Create a clean namespace:

```text
fate_x/acpr_dynflow/
```

Formal trainer:

```text
fate_x.engine.train_acpr_dynflow
```

Formal evaluator:

```text
fate_x.engine.eval_acpr_dynflow
```

The formal import graph must not instantiate or import as formal model components:

```text
fate_x.acpr_flow.model.ACPRFlowModel
fate_x.acpr_flow_v2.model.ACPRFlowCalV2Model
fate_x.models.token_pmt_adapter.TokenPMTAdapter
fate_x.models.sinkhorn_transport.LogSinkhornTransport
fate_x.models.temporal_evidence_memory.TemporalEvidenceMemory
legacy FlowTrace/FlowCal trainers or losses
```

Legacy code may remain for historical reproduction only.

---

# 6. Data and evaluation contract

## 6.1 Formal data paths

```text
train image/text/control:
datasets_part/BDDX/training_32frames.yaml

test image/text:
datasets/BDDX/testing_32frames.yaml

test control:
datasets_part/BDDX/testing_32frames.yaml
```

The loader must verify sample-ID alignment between caption and control data.

## 6.2 Text tensor contract

Use the ADAPT-compatible settings:

```text
32 frames
224 × 224
use_sep_cap = true
action max tokens = 15
explanation max tokens = 15
max sequence length = 30
mask probability = 0.5
max masked tokens = 45
WordPiece tokenizer from local bert-base-uncased
```

Preserve raw action and justification strings in batch metadata.

## 6.3 Signal contract

Implement one `BDDSignalCodec` and use it for:

```text
dataset tensor conversion
training normalization
model output decoding
loss computation
official evaluation
intervention effect normalization
visualization
```

The codec must discover and record:

```text
signal order
raw units
valid/invalid marker
train mean/std
target range
time length
```

Do not assume course is degrees or circular. Primary metrics must reproduce the original ADAPT evaluation behavior exactly. Circular diagnostics may be added only after a data-contract audit confirms they are appropriate, and they must never replace the official metric.

## 6.4 ADAPT baseline evaluation

Before formal training, separately evaluate the user's strongest ADAPT reproduction checkpoint with the exact same:

```text
test sample IDs
caption evaluator
control evaluator
tokenization
beam size
signal order
invalid-value mask
```

This baseline command is comparison-only. Its checkpoint and predictions cannot enter ACPR-DynFlow training.

Write:

```text
adapt_reference_metrics.json
adapt_reference_predictions_manifest.json
```

## 6.5 Primary metrics

Description/action and explanation are evaluated separately with the same ADAPT/COCO metric code:

```text
BLEU-1
BLEU-2
BLEU-3
BLEU-4
METEOR
ROUGE-L
CIDEr
SPICE
```

Control:

```text
RMSE
MAE
Acc@0.1
Acc@0.5
Acc@1
Acc@5
Acc@10
```

for speed and course separately.

The heuristic action-text decision proxy currently found in `acpr_action_text_eval.py` is diagnostic only and must not be used for best selection or headline results.

---

# 7. Typed output contracts

Create `fate_x/acpr_dynflow/types.py`.

## 7.1 Batch

```python
@dataclass
class DynFlowBatch:
    frames: Tensor                    # [B,32,3,224,224]
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    masked_pos: Tensor
    masked_ids: Tensor
    control_target: Tensor | None     # [B,32,2] in codec order
    sample_ids: list[str]
    raw_actions: list[str]
    raw_justifications: list[str]
```

## 7.2 Backbone output

```python
@dataclass
class DynFlowBackboneOutput:
    local_grid: Tensor                # [B,Tp,H,W,Dp]
    coarse_grid: Tensor               # [B,Tp,Hc,Wc,Dp]
    global_sequence: Tensor           # [B,Tg,Dg]
    text_visual_tokens: Tensor        # [B,N,Dtext]
    forward_count: int
```

## 7.3 Predicate output

```python
@dataclass
class DynamicPredicateField:
    names: tuple[str, ...]             # exact 32
    logits: Tensor                    # [B,Tp,32]
    probabilities: Tensor             # [B,Tp,32]
    tokens: Tensor                    # [B,Tp,32,D]
    evidence_maps: Tensor             # [B,Tp,32,H,W]
    confidence: Tensor                # [B,Tp,32]
    centroid: Tensor                  # [B,Tp,32,2]
    relative_centroid_motion: Tensor  # [B,Tp-1,32,2]
    lane_mass: Tensor                 # [B,Tp,32,3]
    query_states: Tensor              # [B,Tp,32,D]
```

## 7.4 Covariate and flow output

```python
@dataclass
class PredicateCovariates:
    raw_covariates: Tensor            # [B,Tp,32,C]
    homogenized: Tensor               # [B,Tp,32,D]
    multiscale: dict[str, Tensor]
    pattern_logits: Tensor            # [B,Tp,32,4]
    pattern_probs: Tensor             # [B,Tp,32,4]
    pattern_names: tuple[str, ...]
```

```python
@dataclass
class TrafficFlowState:
    factor_names: tuple[str, ...]      # exact 13
    factor_tokens: Tensor              # [B,Tp,13,D]
    factor_logits: Tensor              # [B,Tp,13]
    factor_probs: Tensor               # [B,Tp,13]
    lateral_bias: Tensor               # [B,Tp,1], [-1,1]
    factor_to_predicate: Tensor        # [B,Tp,13,32]
    evidence_maps: Tensor              # [B,Tp,13,H,W]
    response_lag_weights: Tensor       # [B,32,13,4]
    lag_aligned_tokens: Tensor         # [B,32,13,D]
    lineage: list[dict]
```

## 7.5 Decision ledger

```python
@dataclass
class DecisionLedger:
    signal_names: tuple[str, ...]
    global_prediction_normalized: Tensor     # [B,32,2]
    factor_contributions_normalized: Tensor # [B,32,13,2]
    final_prediction_normalized: Tensor      # [B,32,2]
    global_prediction_raw: Tensor            # [B,32,2]
    factor_contributions_raw: Tensor         # [B,32,13,2]
    final_prediction_raw: Tensor             # [B,32,2]
    speed_factor_attention: Tensor           # [B,32,13]
    course_factor_attention: Tensor          # [B,32,13]
```

The audit must verify:

```python
final_prediction == global_prediction + factor_contributions.sum(dim=2)
```

within numerical tolerance in both normalized and raw units.

## 7.6 Text output

```python
@dataclass
class DynFlowTextOutput:
    action_logits: Tensor
    explanation_logits: Tensor
    action_loss: Tensor
    explanation_loss: Tensor
    explanation_to_factor_attention: Tensor
    action_to_factor_attention: Tensor
    generated_action: list[str] | None
    generated_explanation: list[str] | None
```

## 7.7 Formal output

```python
@dataclass
class ACPRDynFlowOutput:
    total_loss: Tensor
    loss_components: dict[str, Tensor]
    backbone: DynFlowBackboneOutput
    predicates: DynamicPredicateField
    covariates: PredicateCovariates
    flow: TrafficFlowState
    ledger: DecisionLedger
    text: DynFlowTextOutput
    diagnostics: dict[str, Any]
```

No positional tuple-tail parsing is allowed.

---

# 8. Independent video backbone

File:

```text
fate_x/acpr_dynflow/video_backbone.py
```

Use the repository's Video Swin implementation but create an independent ACPR-DynFlow wrapper.

Initialization:

```text
Kinetics-600 Video Swin checkpoint only
```

Do not load `swin.*` or `fc.*` from ADAPT.

One forward must return:

- a middle spatial stage for named predicates;
- a coarser stage for common ego-camera shift estimation;
- a final temporal sequence for global decisions;
- projected visual tokens for text decoding.

All temporal/spatial dimensions are inferred.

Fixed trainability for all 20 epochs:

```text
Video Swin stages 0–1 frozen
Video Swin stages 2–3 trainable
```

No epoch-based change.

---

# 9. BDD-OIA predicate transfer

Files:

```text
predicate_ontology.py
predicate_transfer.py
configs/acpr_dynflow_predicates.yaml
```

## 9.1 Required source

Codex must locate and record:

```text
FATE-OIA branch: acpr_calalign_v1_2
best complete ACPR-CalAlign checkpoint
predicate ontology YAML
query/prototype tensor keys
source checkpoint SHA256
```

Formal review is blocked if the OIA checkpoint is unresolved.

## 9.2 Exact 32 predicates

The copied ontology must contain the exact names/order from the OIA branch. At minimum the audit checks all known entries such as:

```text
traffic_light_red
traffic_light_green
stop_sign_present
front_vehicle_close
front_vehicle_far
pedestrian_front
cyclist_front
obstacle_front
lane_left_available
lane_right_available
left_turn_region
right_turn_region
drivable_center
drivable_left
drivable_right
road_clear
road_crowded
parked_vehicle_left
parked_vehicle_right
vehicle_left
vehicle_right
traffic_light_visible
traffic_sign_visible
crosswalk_region
intersection_region
merging_left_context
merging_right_context
ego_lane_centered
low_front_visibility
open_left_gap
open_right_gap
global_scene_context
```

## 9.3 Query initialization

\[
q_k^{X}
=
W_{OIA}q_k^{OIA}
+
W_{name}E_{BERT}(name_k)
+
r_k
\]

Requirements:

- OIA query is loaded and detached as initialization prior;
- BERT name embedding is computed once from the local generic BERT;
- residual \(r_k\) is trainable;
- dimensions are mapped explicitly;
- query-source contributions are logged;
- a small semantic anchor loss prevents complete drift.

---

# 10. Calibrated Dynamic Predicate Field

File:

```text
dynamic_predicate_field.py
```

## 10.1 Recurrent semantic queries

For native Video Swin time step \(t\):

\[
q_{t,k}
=
q_k^{X}
+
GRU(e_{t-1,k},q_{t-1,k})
\]

\[
A_{t,k}
=
entmax_{1.5}
\left(
q_{t,k}^{\top}K_t
+
b_k^{region}
\right)
\]

\[
e_{t,k}
=
\sum_{x,y}A_{t,k}(x,y)V_t(x,y)
\]

Use one GRU cell shared across predicates plus predicate embeddings. Avoid 32 separate heavy GRUs.

## 10.2 Region priors

Port the ACPR region-prior semantics. Priors are soft biases, not hard masks.

## 10.3 Lightweight ego-motion compensation

Estimate only a common image-plane shift from coarse features by a bounded local correlation search. Do not perform full patch tracking or Sinkhorn.

Subtract the common shift from predicate centroid displacement:

\[
\Delta \mu_{t,k}^{relative}
=
\Delta \mu_{t,k}
-
g_t
\]

This is an apparent-motion proxy, not metric optical flow.

## 10.4 Confidence

Predicate confidence combines:

```text
presence probability
entmax concentration
temporal query consistency
evidence-feature agreement
visibility/border penalty
```

All components are logged.

---

# 11. Distribution-Calibrated nnPU and Dynamic CalAlign

Files:

```text
nnpu_calalign.py
configs/acpr_dynflow_text_rules.yaml
```

BDD-X text is incomplete. Unmentioned predicates are unlabeled, not negative.

## 11.1 Text rules

YAML defines for each predicate:

```text
positive phrases
contradictory phrases
exclusion phrases
priority
```

Rules must avoid known errors such as matching `light traffic` as `traffic_light`.

## 11.2 nnPU risk

For predicate \(k\):

\[
R_k^{PU}
=
\pi_k R_k^+
+
\max
\left(
0,
R_k^{u-}
-
\pi_k R_k^{+-}
\right)
\]

Requirements:

- positive set from explicit train justification/action evidence;
- reliable negatives only from explicit contradiction;
- all other samples remain unlabeled;
- no fixed unknown-negative BCE;
- per-predicate class prior \(\pi_k\);
- numerically stable nonnegative risk.

## 11.3 Online train-only calibration

During each train epoch, accumulate score histograms for known positive/reliable-negative samples.

At the epoch boundary compute per-predicate:

```text
temperature
positive threshold
negative threshold
class prior
precision/recall estimate on train-known labels
```

No extra image pass and no test labels.

Downstream decision modules use soft probabilities. Calibrated thresholds are used for:

```text
intermediate predicate output
visualization
state confidence
pseudo-positive acceptance
```

The calibration state is saved and resumed.

---

# 12. Predicate covariate homogenization

File:

```text
covariate_homogenizer.py
```

For each predicate/time:

\[
c_{t,k} =
[
p_{t,k},
\Delta p_{t,k},
\mu_{t,k}^{x,y},
\Delta \mu_{t,k}^{relative},
m^L_{t,k},
m^C_{t,k},
m^R_{t,k},
confidence_{t,k}
]
\]

Use a shared MLP plus predicate semantic embedding:

\[
z_{t,k}
=
MLP_{hom}(c_{t,k})
+
W_sE(name_k)
\]

All heterogeneous predicate measurements become a common temporal representation.

---

# 13. Multi-scale Dynamic Pattern Router

File:

```text
multiscale_pattern_router.py
```

Use only three scales:

```text
1× native sequence
2× average-pooled
4× average-pooled
```

Each scale uses:

```text
depthwise temporal convolution
pointwise gated MLP
normalization
```

Fuse coarse-to-fine progressively. Do not use FFT, Mamba, or another full Transformer.

## 13.1 Four traffic patterns

```text
stable
forming
releasing
oscillating
```

\[
\pi_{t,k}
=
softmax(W_r z_{t,k})
\]

\[
\widetilde z_{t,k}
=
z_{t,k}
+
\sum_{b=1}^{4}
\pi_{t,k,b}M_b(z_{t,k})
\]

Use low-rank pattern adapters.

## 13.2 Weak pattern semantics

Build detached soft targets from smoothed first/second temporal differences:

- positive trend → forming;
- negative trend → releasing;
- low first/second difference → stable;
- high alternating second difference → oscillating.

Use a low-weight consistency loss. Pattern names must therefore have operational meaning.

---

# 14. Ego-centric mesoscopic lane/corridor flow

File:

```text
mesoscopic_lane_flow.py
```

Construct soft left/center/right corridor masks from:

```text
geometry prior
drivable_left/center/right
lane-left/right availability
ego-lane-centered predicate
```

Using only dynamic participant predicates, compute per corridor/time:

```text
visual occupancy proxy
relative mobility
motion coherence
stopped tendency
queue pressure
gap openness
```

Do not call these physical density or metric velocity.

The left/center/right values must be distinct tensors. Replicating one global motion statistic across all three corridors is forbidden.

---

# 15. Compositional traffic-state reasoner

File:

```text
traffic_state_reasoner.py
```

## 15.1 Thirteen factors

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

Output an additional signed lateral bias in `[-1,1]`.

## 15.2 State construction

Each factor query sparsely attends to:

```text
dynamic predicate covariates
corridor-flow tokens
pattern routing outputs
semantic-name embedding
grammar support/contradiction prior
```

Use entmax over predicates/corridors.

## 15.3 Low-rank compositional interaction

Represent regime/phase/source jointly:

\[
f_t =
W_Rr_t + W_Pp_t + W_Ss_t
+
\sum_h
(u_h^\top r_t)
(v_h^\top p_t)
(w_h^\top s_t)d_h
\]

This permits unseen combinations without defining a large flat class set.

## 15.4 Evidence lineage

For every factor store:

```text
predicate support weights
corridor support weights
pattern probabilities
spatial evidence map
confidence
```

---

# 16. Response-Lag Alignment

File:

```text
response_lag.py
```

Traffic events and ego response need not be simultaneous.

For each 32-step decision query, factor, and lag \(\ell\in\{0,1,2,3\}\):

\[
\lambda_{t,k,\ell}
=
softmax_{\ell}
(h_t^\top W_{\ell}f_{t-\ell,k})
\]

\[
\widehat f_{t,k}
=
\sum_{\ell=0}^{3}
\lambda_{t,k,\ell}f_{t-\ell,k}
\]

Requirements:

- causal indexing;
- out-of-range lags masked;
- lag weights sum to one;
- lag-disabled mode available for audit;
- synthetic delayed-response test recovers the known lag.

---

# 17. Independent global decision stream

Files:

```text
global_decision_stream.py
signal_codec.py
```

## 17.1 Global decision encoder

From final Video Swin temporal features:

```text
two lightweight temporal mixing blocks
32 learned output-time queries
cross-attention from output queries to video sequence
```

Output:

```text
global decision tokens [B,32,D]
global normalized speed/course [B,32,2]
```

This stream is entirely independent of ADAPT.

## 17.2 Signal codec

The model trains in standardized train-split units and decodes to raw units before official evaluation.

The codec maps signal indices by names and never assumes speed/course position from memory.

Primary official control error uses the exact ADAPT evaluator contract.

---

# 18. Factor-Decomposed Decision Ledger

File:

```text
decision_ledger.py
```

## 18.1 Factor-specific contributions

For each output time and factor:

\[
\Delta s_{t,k}
=
a_{t,k}^{s}\delta_{t,k}^{s}
\]

\[
\Delta c_{t,k}
=
a_{t,k}^{c}\delta_{t,k}^{c}
\]

Speed reader has a semantic prior toward longitudinal factors. Course reader has a prior toward directional/lateral factors. Priors are soft biases and can be corrected by data.

## 18.2 Final prediction

\[
s_t =
s_t^{global}
+
\sum_k \Delta s_{t,k}
\]

\[
c_t =
c_t^{global}
+
\sum_k \Delta c_{t,k}
\]

## 18.3 Complementarity objective

Train both:

```text
global prediction against GT
final prediction against GT
flow contribution sum against stopgrad(GT - global prediction)
```

This prevents the global stream from eliminating the interpretable flow stream and avoids requiring a separate global-only training run.

## 18.4 Exact ledger

Every saved state contribution must be the exact tensor added to the final prediction. No gradient-based attribution is substituted.

---

# 19. Independent action/explanation decoder

Files:

```text
text_decoder.py
contribution_alignment.py
```

Initialize from generic local BERT-base, not an ADAPT task checkpoint.

## 19.1 Inputs

Action segment reads:

```text
global decision tokens
traffic-state summary
```

Explanation segment reads:

```text
traffic factor tokens
predicate support tokens
signed decision-contribution tokens
action hidden states
detached global decision tokens
```

## 19.2 Gradient direction

Explanation may update:

```text
predicate field
covariate homogenizer
traffic-state reasoner
text decoder
```

Explanation must not update the global decision stream:

```python
explanation_global_context = global_decision_tokens.detach()
```

This structural separation replaces complex PCGrad/firewall logic.

## 19.3 Separate losses

Compute token losses separately by segment:

```text
action_text_loss
explanation_text_loss
```

A single duplicated caption loss is forbidden.

## 19.4 Contribution alignment

Aggregate the real absolute state contributions:

\[
q_k =
Normalize_t
\left(
|\Delta s_{t,k}|/\sigma_s
+
|\Delta c_{t,k}|/\sigma_c
\right)
\]

Align explanation-to-state attention:

\[
L_{align}
=
JS
\left(
A^{exp\rightarrow state},
q
\right)
\]

The explanation therefore references states that actually affect vehicle decisions.

## 19.5 Inference

No GT action or justification may enter inference. Generated action can condition generated explanation through decoder hidden states.

---

# 20. Losses

File:

```text
fate_x/losses/acpr_dynflow_losses.py
```

All losses must be separately logged with raw value, weighted value, and intended gradient targets.

Recommended fixed weights:

```text
final_speed_normalized          1.00
final_course_normalized         1.00
global_speed_normalized         0.50
global_course_normalized        0.50
flow_residual_speed             0.25
flow_residual_course            0.25
control_first_difference        0.20

action_text                     0.35
explanation_text                0.50

predicate_nnpu                  0.10
predicate_query_anchor          0.02
pattern_semantic                0.05
traffic_grammar                 0.02
contribution_alignment          0.05
temporal_consistency            0.02
contribution_sparsity           0.01
contribution_smoothness         0.01
```

Use normalized Huber/SmoothL1 for control.

Do not use in the formal run:

```text
HardPair
uncertainty task weighting
SCST
global Sinkhorn
Gradient Firewall/PCGrad
unknown-as-negative BCE
ADAPT distillation
```

---

# 21. One-run training configuration

Formal training is one unified 20-epoch run.

```text
epochs: 20
precision: BF16
optimizer: AdamW
scheduler: 5% linear warm-up + cosine decay
gradient clip: 1.0
gradient checkpointing: enabled
metric early stopping: disabled
```

Fixed trainability throughout all epochs:

```text
Video Swin stages 0–1 frozen
Video Swin stages 2–3 trainable
BERT bottom 8 layers frozen
BERT top 4 layers trainable
all ACPR-DynFlow modules trainable
```

Learning rates:

```text
Video Swin stages 2–3             1.0e-5
predicate transfer/query/GRU      1.0e-4
nnPU/calibration score heads      5.0e-5
homogenizer/pattern router        1.0e-4
traffic-state reasoner            1.0e-4
response-lag module               5.0e-5
global decision stream            1.0e-4
decision ledger                   7.5e-5
BERT top 4                        2.0e-5
text state/contribution adapters  5.0e-5
LM head                           2.0e-5
```

Weight decay:

```text
new weights: 0.01
backbone weights: 0.05
bias/norm/gates: 0
```

Contribution scales may ramp from 0.10 to 1.00 during the first 5% of optimizer steps. This is initialization warm-up, not staged training.

---

# 22. RTX 5880 48 GiB memory policy

Target measured peak reserved memory:

```text
preferred: 42–46 GiB
hard limit: 46.0 GiB
minimum safety headroom: approximately 1.5 GiB
```

Do not allocate dummy tensors.

Probe in order:

```text
micro batch 10 / accumulation 7  -> effective 70
micro batch 8  / accumulation 8  -> effective 64
micro batch 6  / accumulation 11 -> effective 66
micro batch 5  / accumulation 13 -> effective 65
micro batch 4  / accumulation 16 -> effective 64
micro batch 3  / accumulation 22 -> effective 66
micro batch 2  / accumulation 32 -> effective 64
```

For every candidate:

```text
3 warm-up iterations
30 measured complete forward/backward/optimizer-simulation iterations
direct 32-frame images
all formal losses
BF16
no cache
```

Select the largest stable micro-batch satisfying:

```text
peak reserved <= 46.0 GiB
no OOM
no NaN/Inf
no skipped optimizer step
```

Record allocated/reserved peak, step time, image throughput, and effective batch.

---

# 23. Per-epoch evaluation and checkpoint selection

At the end of every epoch:

1. atomically save `checkpoint_latest.pth`;
2. evaluate full test text with the ADAPT-compatible evaluator;
3. evaluate full test continuous control with the ADAPT-compatible evaluator;
4. evaluate a fixed 256-test-sample lightweight traffic-flow audit;
5. save metrics and checkpoints;
6. continue to the next epoch regardless of metric direction.

No validation loader.

Save:

```text
checkpoint_best_text.pth
checkpoint_best_control.pth
checkpoint_best_joint.pth
checkpoint_best_test.pth
checkpoint_latest.pth
```

## 23.1 Best text

Maximize:

```text
CIDEr_description + CIDEr_explanation
```

## 23.2 Best control

Minimize:

\[
0.5 \frac{RMSE_{speed}}{RMSE_{speed}^{ADAPT}}
+
0.5 \frac{RMSE_{course}}{RMSE_{course}^{ADAPT}}
\]

using the measured ADAPT reproduction reference.

## 23.3 Best test

Vehicle-decision-first, text-safe lexicographic selector:

Eligibility:

```text
CIDEr_description + CIDEr_explanation
>= 0.85 × measured ADAPT CIDEr sum
```

Among eligible checkpoints:

1. minimize normalized control score;
2. higher explanation CIDEr;
3. higher description CIDEr;
4. lower speed RMSE;
5. lower course RMSE.

If no checkpoint satisfies the text floor:

1. choose minimum normalized control score;
2. record `text_floor_not_met=true`.

Write the full comparison tuple. Do not hide selection in an arbitrary mixed-scale scalar.

---

# 24. Traffic-flow influence verification

The required additional evaluation is model-level traffic-flow dependence.

## 24.1 Lightweight per-epoch audit

On a fixed 256-sample test subset:

```text
full model
all traffic-flow states off
temporal order shuffled
top factor off
```

Record normalized speed/course changes and text-logit changes.

This audit is not used for best selection.

## 24.2 Full best-checkpoint audit

On `checkpoint_best_test.pth`, run the full test set:

```text
all-flow-off
regime-off
phase-off
source-off
individual top-factor-off
predicate-off
evidence-tube-off
equal-mass random evidence deletion, 5 seeds
temporal shuffle
temporal reverse
lag disabled
last-frame-only
```

Every intervention must rerun all downstream components from the earliest affected layer.

## 24.3 Required metrics

### Traffic Flow Utilization

\[
TFU_s =
RMSE_s(flow\ off)-RMSE_s(full)
\]

\[
TFU_c =
RMSE_c(flow\ off)-RMSE_c(full)
\]

### Evidence Specificity

\[
ES =
\Delta Error_{evidence}
-
E[\Delta Error_{random\ equal\ mass}]
\]

### Temporal Direction Reliance

\[
TDR =
Error(reverse/shuffle)-Error(full)
\]

### Lag Necessity

\[
LN =
Error(lag=0)-Error(learned\ lag)
\]

### Contribution Fidelity

Rank factors by exact ledger contribution and by factor-off effect; report Spearman correlation.

### Text–Decision Consistency

Compare explanation-to-factor attention with normalized exact decision contributions.

Report results on:

```text
all samples
traffic-signal subset
queue/following subset
turn-left subset
turn-right subset
merge/lane-change subset
vulnerable-road-user subset
```

Describe these as model-level dependence, not real-world causal effect.

---

# 25. Dynamic Traffic Decision Ledger visualization

Create:

```text
fate_x/explain/acpr_dynflow_renderer.py
fate_x/explain/acpr_dynflow_atlas.py
```

Each selected sample must contain:

1. **Predicate Evidence Tubes**
   - 8 keyframes;
   - stable color per named predicate;
   - real evidence maps over de-normalized frames.

2. **Mesoscopic Flow Ribbons**
   - left/center/right occupancy proxy;
   - relative mobility;
   - stopped tendency;
   - queue pressure;
   - gap openness.

3. **Pattern and State Lattice**
   - stable/forming/releasing/oscillating;
   - regime/phase/source;
   - lateral bias;
   - confidence;
   - support predicates.

4. **Event–Response Lag Ribbon**
   - event activation;
   - selected lag;
   - resulting speed/course contribution.

5. **Signed Decision Waterfall**
   - global contribution;
   - each traffic-state contribution;
   - exact final speed/course reconstruction.

6. **Contribution-Aligned Text**
   - baseline model action/explanation;
   - generated action/explanation;
   - token-to-state attention;
   - state contribution values.

7. **Counterfactual Twin**
   - factual output;
   - state-off;
   - evidence-off;
   - nearest feasible train-state replacement;
   - validity, distance, modified-factor count, temporal plausibility.

Dataset atlas:

```text
regime/phase/source → action matrix
factor → speed effect matrix
factor → course effect matrix
response-lag distribution
state transition graph
explanation phrase → contribution factor
evidence vs random deletion distributions
global vs flow decision coverage
failure cases
```

Generate PNG/JSON for cases and a standalone HTML atlas.

No manual boxes and no fabricated values.

---

# 26. File-level implementation plan

## 26.1 Add

```text
configs/acpr_dynflow_v1_bddx_32f_224.yaml
configs/acpr_dynflow_predicates.yaml
configs/acpr_dynflow_text_rules.yaml
configs/acpr_dynflow_traffic_grammar.yaml

fate_x/acpr_dynflow/__init__.py
fate_x/acpr_dynflow/config.py
fate_x/acpr_dynflow/types.py
fate_x/acpr_dynflow/signal_codec.py
fate_x/acpr_dynflow/video_backbone.py
fate_x/acpr_dynflow/predicate_ontology.py
fate_x/acpr_dynflow/predicate_transfer.py
fate_x/acpr_dynflow/ego_motion.py
fate_x/acpr_dynflow/dynamic_predicate_field.py
fate_x/acpr_dynflow/nnpu_calalign.py
fate_x/acpr_dynflow/covariate_homogenizer.py
fate_x/acpr_dynflow/multiscale_pattern_router.py
fate_x/acpr_dynflow/mesoscopic_lane_flow.py
fate_x/acpr_dynflow/traffic_state_reasoner.py
fate_x/acpr_dynflow/response_lag.py
fate_x/acpr_dynflow/global_decision_stream.py
fate_x/acpr_dynflow/decision_ledger.py
fate_x/acpr_dynflow/contribution_alignment.py
fate_x/acpr_dynflow/text_decoder.py
fate_x/acpr_dynflow/interventions.py
fate_x/acpr_dynflow/model.py

fate_x/losses/acpr_dynflow_losses.py

fate_x/engine/acpr_dynflow_data.py
fate_x/engine/eval_adapt_reference.py
fate_x/engine/train_acpr_dynflow.py
fate_x/engine/eval_acpr_dynflow.py
fate_x/engine/run_acpr_dynflow_preflight.py
fate_x/engine/audit_acpr_dynflow.py
fate_x/engine/probe_acpr_dynflow_memory.py
fate_x/engine/export_acpr_dynflow_visuals.py
fate_x/engine/build_acpr_dynflow_atlas.py
fate_x/engine/supervise_acpr_dynflow_foreground.py

fate_x/explain/acpr_dynflow_renderer.py
fate_x/explain/acpr_dynflow_atlas.py
fate_x/explain/acpr_dynflow_faithfulness.py

scripts/FATE_X_acpr_dynflow_v1_foreground.ps1
scripts/FATE_X_acpr_dynflow_v1_foreground.sh

docs/runbooks/ACPR_DynFlow_V1_Implementation_Plan.md
docs/runbooks/ACPR_DynFlow_V1_Implementation_Manifest.json
.codex/skills/acpr-dynflow-implementation-audit/SKILL.md
```

## 26.2 Modify only when required

```text
src/modeling/load_swin.py
src/modeling/video_swin/swin_transformer.py
src/layers/bert/modeling_bert.py
src/datasets/vision_language_tsv.py
src/datasets/vl_dataloader.py
src/datasets/caption_tensorizer.py
.gitignore
```

Shared-file modifications must remain backward compatible with original ADAPT and old branches.

Do not use `src/tasks/run_adapt.py` as the formal ACPR-DynFlow trainer.

---

# 27. Required tests

Create `tests/acpr_dynflow/`.

## Contracts and isolation

```text
test_formal_import_graph.py
test_config_binding.py
test_named_outputs.py
test_no_adapt_checkpoint_in_training.py
test_direct_image_no_cache.py
```

## Data and metrics

```text
test_adapt_text_tensor_contract.py
test_signal_codec_contract.py
test_adapt_metric_parity.py
test_sample_id_alignment.py
test_test_only_protocol.py
```

## OIA transfer and predicates

```text
test_exact_32_predicates.py
test_oia_query_transfer.py
test_region_priors.py
test_recurrent_predicate_field.py
test_entmax_evidence.py
test_ego_motion_compensation.py
test_predicate_temporal_order.py
```

## PU and calibration

```text
test_text_rules.py
test_nnpu_nonnegative_risk.py
test_unknown_not_negative.py
test_predicate_prior_estimation.py
test_online_calalign_train_only.py
test_calibration_resume.py
```

## Covariates and temporal reasoning

```text
test_covariate_homogenizer.py
test_multiscale_pattern_router.py
test_pattern_semantics.py
test_lane_flow_distinct_regions.py
test_traffic_state_composition.py
test_state_evidence_lineage.py
test_response_lag.py
```

## Decision and text

```text
test_signal_global_stream.py
test_factor_ledger_exact_sum.py
test_speed_course_factor_routing.py
test_flow_residual_target.py
test_separate_action_explanation_losses.py
test_explanation_detaches_global_decision.py
test_contribution_alignment.py
test_inference_no_gt_text.py
```

## Training and artifacts

```text
test_optimizer_groups.py
test_scheduler_resume.py
test_best_selectors.py
test_atomic_checkpoint.py
test_intervention_recompute.py
test_equal_mass_random.py
test_visual_canvas_schema.py
test_atlas_schema.py
test_foreground_supervisor.py
test_review_pass_binding.py
test_e2e_direct_image_smoke.py
```

---

# 28. Pre-training gates

Formal training is blocked until every gate passes.

## Gate A — Git and import graph

- correct new worktree/branch;
- clean;
- local SHA equals GitHub SHA;
- formal trainer imports only ACPR-DynFlow formal modules;
- no unresolved config paths.

## Gate B — Data/evaluation parity

- same test sample IDs as ADAPT;
- text tensorizer matches;
- signal order/units/mask verified;
- running the new metric bridge on saved ADAPT predictions reproduces the original ADAPT JSON.

## Gate C — Model independence

- no ADAPT task checkpoint loaded in ACPR-DynFlow;
- Kinetics and BERT generic weights load;
- OIA predicate checkpoint loads;
- no ADAPT prediction file read.

## Gate D — Static/unit suite

```bash
python -m compileall -q fate_x src
python -m pytest tests/acpr_dynflow -q
python -m pytest tests -q
```

## Gate E — Real direct-image smoke

At least:

```text
8 train samples
8 test samples
8 real optimizer steps
batch 1
beam 1
```

Require all outputs/losses/artifacts finite and checkpoint/evaluation complete.

## Gate F — Gradient chain

On real images require finite nonzero gradients for:

```text
OIA query mapper and predicate residual
predicate GRU
predicate visual projections
homogenizer
all pattern bases and router
lane-flow encoder
traffic-state queries
response-lag parameters
global decision stream
speed ledger
course ledger
BERT top layers
action decoder adapter
explanation decoder adapter
```

Frozen modules must have no gradients.

## Gate G — 128-sample mechanism fit

Use a deterministic 128-train-sample subset and bounded updates.

Require:

```text
final control loss decreases
global control loss decreases
flow residual loss decreases
action loss decreases
explanation loss decreases
nnPU known-label risk decreases
pattern predictions do not collapse
traffic factors do not become constant
flow contribution is nonzero
course uses directional factors on lateral samples
```

This is not a performance claim.

## Gate H — Temporal/lag necessity

On synthetic and real samples:

```text
reverse changes forming/releasing
shuffle changes pattern/state
lag recovery finds known delayed response
lag disabled changes decision output
last-frame-only differs from full sequence
```

## Gate I — Real influence smoke

Require:

```text
flow-off changes actual speed or course
top-factor-off changes its exact contribution
evidence-off changes downstream states and decisions
equal-mass random control is matched
```

## Gate J — Visualization

Render one real complete ledger canvas and mini atlas.

## Gate K — Memory probe

Select stable configuration under the 46 GiB cap.

## Gate L — Independent audit

Run the audit skill on the exact clean pushed SHA.

---

# 29. Review pass

Required:

```text
.background_runs/acpr_dynflow_v1_preflight/
REVIEW_PASS_ACPR_DYNFLOW_V1.txt
```

The pass binds:

```text
source SHA
new branch SHA
GitHub SHA
plan/config/skill hashes
all gate reports
selected memory configuration
```

Any code/config/script/test change invalidates the pass.

---

# 30. Foreground supervisor contract

The supervisor sequence is:

```text
verify review pass and SHA
evaluate ADAPT reference
run final memory probe
launch 20-epoch ACPR-DynFlow training
evaluate full test after every epoch
maintain checkpoint files
run final full traffic-flow audit on best-test
render Canvas and Atlas
update canonical records
verify final GitHub SHA
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

Required behavior:

- synchronous attached child process;
- live stdout/stderr;
- 60-second heartbeat even during evaluation;
- no stop because metrics are poor;
- OOM fallback to next audited batch while preserving effective batch;
- retry transient I/O/evaluator errors up to three times;
- reject `.tmp` checkpoints;
- resume atomic `checkpoint_latest.pth`;
- explicit user stop sentinel:
  ```text
  <run>/control/STOP_REQUESTED_BY_USER
  ```
- never create that sentinel without explicit user instruction.

For a reproducible code defect:

```text
preserve log/checkpoint
systematic debugging
write regression test
fix
commit
push
invalidate old review pass
rerun full audit
resume latest
continue foreground
```

Normal termination is permitted only when:

1. the complete configured suite finishes;
2. the user explicitly requests stop;
3. unrecoverable hardware/OS failure requires user intervention.

---

# 31. Formal experiment command

After review pass:

```powershell
Set-Location E:\sbw\FATE_Drive\fate_x_acpr_dynflow_v1_worktree

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_X_acpr_dynflow_v1_foreground.ps1 `
  -Config configs\acpr_dynflow_v1_bddx_32f_224.yaml `
  -RequireReviewPass
```

The launcher remains attached.

---

# 32. Required epoch artifacts

Each epoch:

```text
epoch_XXX/
  description_metrics.json
  explanation_metrics.json
  control_metrics.json
  loss_components.jsonl
  gradient_norms.json
  predicate_calibration.json
  predicate_stats.json
  pattern_stats.json
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
signal_contract.json
oia_transfer_report.json
optimizer_groups.json
memory_probe.json
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

No fake zero values. Missing values require an explicit unavailable reason.

---

# 33. Definition of implementation complete

Codex may state implementation complete only when:

- the new worktree and branch exist and are pushed;
- the formal import graph is independent;
- exact 32 OIA predicates and source checkpoint are loaded;
- direct-image/no-cache training is proven;
- data and metric parity with ADAPT is proven;
- every component executes and has intended gradients;
- signal codec is verified;
- decision ledger reconstructs final control exactly;
- action/explanation losses are separate;
- explanation uses real decision contributions;
- interventions rerun real downstream computations;
- a real complete Canvas/Atlas smoke exists;
- all tests and mechanism gates pass;
- the review pass binds the exact clean pushed SHA.

File existence or unit tests alone are insufficient.

---

# 34. Definition of formal experiment complete

The experiment is complete only after:

- ADAPT reference evaluation is recorded;
- all 20 ACPR-DynFlow epochs finish;
- full test is evaluated after each epoch;
- latest and all best checkpoints exist;
- final best-test full traffic-flow audit finishes;
- complete Canvas/Atlas exports exist;
- final local/GitHub SHAs match;
- canonical task/findings/progress ledgers are updated;
- `run_complete.json` records successful completion.

---

# 35. Hard failures

Do not authorize training if any is true:

```text
formal model loads ADAPT task weights
formal trainer imports V1/V2 ACPR model
predicate names are anonymous
OIA query checkpoint is not loaded
unknown text is treated as a hard negative
pattern names have no operational target
left/center/right flow values are replicated
response lag is configured but unused
global and flow decision terms do not exactly sum to final control
speed/course use the same undifferentiated factor reader
course signal order/unit is assumed rather than audited
action/explanation share one duplicated loss
explanation gradients reach the global decision stream
decision contributions are post-hoc approximations
intervention only changes a display tensor
ADAPT metric parity is not proven
feature/token/logit cache is read
a validation loader is created
test labels fit model/calibration parameters
memory is filled with dummy allocations
supervisor detaches or metric-stops
worktree is dirty
local and GitHub SHAs differ
review pass belongs to another SHA
```
