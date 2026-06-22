# ACPR-FlowCal++ V2 on BDD-X
## Codex code-level implementation, audit, experiment, and foreground-supervision contract

**Target repository:** `d2116056543-lab/FATE-X`
**Target worktree:** `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree`
**Target branch:** `flowtrace_pmt_v1`
**No new worktree. No new branch. Modify the current worktree and push the same branch.**

---

## 0. Non-negotiable objective

Implement a formal BDD-X method that preserves the successful BDD-OIA ACPR principle:

```text
named visual predicates
→ explicit intermediate driving semantics
→ reason-mediated decision
```

and extends it to video as:

```text
32 direct RGB frames
→ transported named predicate trajectories
→ lane-wise mesoscopic traffic-flow field
→ regime / phase / source / axis / direction states
→ semantic reason memory
→ protected action/explanation generation
→ longitudinal speed and lateral course mediation
→ intervention-based faithfulness and hierarchical visualization
```

The formal output targets remain the original BDD-X targets:

- action/description text;
- justification/explanation text;
- continuous speed trajectory;
- continuous course trajectory.

No new human annotation is required. Traffic-flow states are learned intermediate variables supervised only by original train text/control signals, weak text rules, transport consistency, and task losses.

The implementation is not complete merely because classes or loss names exist. Every enabled component must:

1. execute on the formal direct-image path;
2. consume its intended inputs;
3. change an intended downstream output under controlled intervention;
4. receive finite, nonzero gradients in the stage where it is trainable;
5. emit auditable tensors and metadata;
6. be covered by unit, integration, real-data smoke, and mechanism tests.

Formal training is forbidden before the V2 audit writes an exact-commit review pass.

---

## 1. Why the current V1 formal path must not be incrementally patched

Codex must first preserve a safety snapshot, then build a separate V2 formal import graph. The following V1 behavior must not survive in V2.

### 1.1 Stage plan is descriptive rather than executable

The current trainer constructs and writes an experiment suite, but its main training loop does not apply the declared stage-specific freeze lists, gate maxima, HardPair start epoch, transport switch, or flow switch. V2 must have an executable `StageController`; YAML-only scheduling is rejected.

### 1.2 The current continuous-control path is not the ADAPT motion baseline

The current ACPR model synthesizes a temporal hidden sequence from a pooled reason state and predicts control with local linear layers. V2 must instead load and execute ADAPT's released `sensor_pred_head` motion transformer and expose:

```text
control_base_prediction
control_hidden
```

Traffic-flow reasoning may only add a bounded residual to that real ADAPT base.

### 1.3 Local transport is calculated but not used to propagate predicate evidence

V2 must use the transport distribution to warp the previous predicate map into the current frame and combine it with the current-frame visual score. Merely logging transport tensors is insufficient.

### 1.4 Equal averaging destroys semantic structure

The current global reason state is an equal average of memory tokens. V2 must use typed, confidence-aware, query-based aggregation and separate longitudinal/lateral memory views.

### 1.5 Current text settings drift from the ADAPT contract

V2 must resolve the caption tensorizer contract from the released baseline metadata when available and otherwise use the official ADAPT BDD-X settings:

```text
mask_prob = 0.5
max_masked_tokens = 45
max_seq_a_length = 15
max_seq_length = 30
use_sep_cap = true
```

The resolved values must be written into the run manifest. Hardcoded `0.15/20` is forbidden for the formal run.

### 1.6 Scheduler and optimizer stages must be real

A YAML field saying `scheduler: cosine` is insufficient. V2 must instantiate, step, save, restore, and audit the scheduler. Trainable parameter sets and learning rates must change at stage boundaries exactly as specified.

### 1.7 Sequence-CalAlign and interventions must be executable

V2 cannot merely expose classes or write optimistic JSON. Calibration must fit all four branches on deterministic train-calib samples. Every intervention must rerun all downstream computations and measure actual text/control deltas.

### 1.8 Existing renderer and atlas are placeholders

A single grayscale map or JSON record dump is not the required visualization. V2 must produce a hierarchical, temporal, decision-linked canvas and a dataset-level atlas.

---

## 2. Repository and Git safety procedure

Codex must execute these steps in the current worktree.

### 2.1 Read durable context first

Read in full:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md

docs/acpr_flowcalpp/ACPR_FlowCalPP_task_plan.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_findings.md
docs/acpr_flowcalpp/ACPR_FlowCalPP_progress.md

docs/runbooks/ACPR_FlowCal_V2_Implementation_Plan.md
.codex/skills/acpr-flowcal-v2-implementation-audit/SKILL.md
configs/acpr_flowcal_v2_bddx_32f_224.yaml
```

Training/experiment status remains recorded only in the established task/findings/progress files. The implementation plan and audit skill are non-status runbook artifacts.

### 2.2 Assert location and branch

Required:

```text
pwd / current Windows path:
E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree

git branch --show-current:
flowtrace_pmt_v1
```

Do not create another worktree or branch.

### 2.3 Preserve pre-V2 state without destructive commands

Run:

```text
git status --short
git diff --stat
git diff --check
git rev-parse HEAD
git ls-remote github refs/heads/flowtrace_pmt_v1
```

If dirty:

1. inspect every changed/untracked file;
2. separate ignored run artifacts from source;
3. save a patch under ignored `.background_runs/v2_preimplementation_snapshot/`;
4. test the current state;
5. commit a clearly named safety snapshot;
6. push `github flowtrace_pmt_v1`;
7. verify local and remote SHA equality.

Forbidden:

```text
git reset --hard
git clean -fd
discarding unknown edits
overwriting another process's changes
```

### 2.4 Commit discipline

Use small coherent commits:

1. plan/config/skill;
2. typed contracts and baseline wrappers;
3. transport and predicate tracker;
4. lane flow and axis-aware state;
5. reason target/memory/firewall;
6. text/control integration;
7. loss/stage/trainer;
8. intervention/evaluation/visuals;
9. audit/preflight/supervisor;
10. fixes found by independent review.

After each commit:

```text
git diff --check
pytest relevant tests
git push github flowtrace_pmt_v1
verify SHA
```

---

## 3. New formal import graph

Create a clean V2 namespace. Legacy V1 modules may remain for historical evaluation, but the V2 trainer must not import them.

```text
fate_x/acpr_flow_v2/
    __init__.py
    config.py
    types.py
    adapt_video_backbone.py
    adapt_motion_backbone.py
    local_partial_transport.py
    temporal_predicate_tracker.py
    lane_flow_field.py
    axis_aware_flow_composer.py
    contextual_reason_target.py
    pu_targets.py
    semantic_reason_memory.py
    semantic_gradient_firewall.py
    temporal_seca.py
    axis_aware_control_adapter.py
    temporal_hardpair.py
    prefix_future.py
    sequence_calalign.py
    interventions.py
    model.py

fate_x/losses/
    acpr_flowcal_v2_losses.py
    explanation_scst.py

fate_x/engine/
    acpr_flowcal_v2_data.py
    train_acpr_flowcal_v2.py
    eval_acpr_flowcal_v2.py
    run_acpr_flowcal_v2_preflight.py
    audit_acpr_flowcal_v2.py
    probe_acpr_flowcal_v2_memory.py
    supervise_acpr_flowcal_v2_foreground.py
    export_acpr_flowcal_v2_visuals.py
    build_acpr_flowcal_v2_atlas.py

fate_x/explain/
    acpr_flowcal_v2_renderer.py
    acpr_flowcal_v2_atlas.py
    acpr_flowcal_v2_faithfulness.py

scripts/
    FATE_X_acpr_flowcal_v2_foreground.ps1
    FATE_X_acpr_flowcal_v2_foreground.sh

tests/acpr_flowcal_v2/
    ...
```

Formal entrypoints:

```text
python -m fate_x.engine.train_acpr_flowcal_v2
python -m fate_x.engine.eval_acpr_flowcal_v2
python -m fate_x.engine.run_acpr_flowcal_v2_preflight
python -m fate_x.engine.audit_acpr_flowcal_v2
python -m fate_x.engine.supervise_acpr_flowcal_v2_foreground
```

The V2 import graph must reject direct imports from:

```text
fate_x.acpr_flow.model
fate_x.models.token_pmt_adapter
fate_x.models.sinkhorn_transport
fate_x.losses.flowtrace_losses
fate_x.explain.flowtrace_renderer
```

Reusing a low-level utility is allowed only after its behavior is tested and the V2 module owns the formal API.

---

## 4. Typed tensor contracts

Implement dataclasses in `fate_x/acpr_flow_v2/types.py`. Do not use positional tail tuples.

### 4.1 Batch

```python
@dataclass
class FlowCalV2Batch:
    input_ids: Tensor | None
    attention_mask: Tensor | None
    token_type_ids: Tensor | None
    frames: Tensor                    # [B,32,3,224,224]
    masked_pos: Tensor | None
    masked_ids: Tensor | None
    car_info: Tensor | None           # [B,2,32]
    sample_ids: list[str]
    raw_actions: list[str]
    raw_justifications: list[str]
```

### 4.2 Backbone output

```python
@dataclass
class VideoBackboneOutput:
    fine_native: Tensor               # [B,Tf,Hf,Wf,D]
    coarse_native: Tensor             # [B,Tc,Hc,Wc,D]
    fine_aligned: Tensor              # [B,32,Hf,Wf,D]
    coarse_aligned: Tensor            # [B,32,Hc,Wc,D]
    fused_grid: Tensor                # [B,32,Hf,Wf,D]
    dense_tokens_raw: Tensor          # original ADAPT Video-Swin tokens
    dense_tokens_projected: Tensor    # ADAPT fc projected tokens
    forward_count: int
```

Only predicate/flow reasoning uses temporally aligned multiscale grids. ADAPT text and motion baselines continue using the original dense projected tokens.

### 4.3 Predicate trajectory

```python
@dataclass
class PredicateTrajectory:
    names: tuple[str, ...]             # exact 32
    attention: Tensor                  # [B,32,32,H,W]
    tokens: Tensor                     # [B,32,32,D]
    presence_logits: Tensor            # [B,32,32]
    presence_probs: Tensor             # [B,32,32]
    confidence: Tensor                 # [B,32,32]
    relative_motion: Tensor            # [B,31,32,2]
    descriptor: Tensor                 # [B,32,D]
    descriptor_parts: dict[str, Tensor]
```

### 4.4 Lane-wise flow field

Use regions ordered `left, center, right`.

```python
@dataclass
class LaneFlowFieldOutput:
    region_names: tuple[str, str, str]
    soft_masks: Tensor                 # [B,32,3,H,W]
    occupancy: Tensor                  # [B,32,3]
    relative_motion: Tensor            # [B,32,3,2]
    motion_coherence: Tensor           # [B,32,3]
    stopped_tendency: Tensor           # [B,32,3]
    queue_pressure: Tensor             # [B,32,3]
    temporal_tokens: Tensor            # [B,32,3,D]
    descriptor: Tensor                 # [B,3,D]
```

These values are explicitly named **ego-centric mesoscopic visual-flow descriptors**, not physical traffic density or metric velocity.

### 4.5 Axis-aware flow state

Keep the original 13 semantic factors and add typed attributes.

```text
semantic factors: 13
lane tokens:       3
axis tokens:       2  (longitudinal, lateral)
direction tokens:  3  (left, neutral, right)
```

```python
@dataclass
class AxisAwareFlowOutput:
    semantic_names: tuple[str, ...]    # exact original 13
    semantic_tokens: Tensor            # [B,13,D]
    semantic_logits: Tensor            # [B,13]
    semantic_probs: Tensor             # [B,13]
    semantic_evidence: Tensor          # [B,32,13,H,W]

    lane_tokens: Tensor                # [B,3,D]
    axis_tokens: Tensor                # [B,2,D]
    axis_logits: Tensor                # [B,2]
    axis_probs: Tensor                 # [B,2]
    direction_tokens: Tensor           # [B,3,D]
    direction_logits: Tensor           # [B,3]
    direction_probs: Tensor            # [B,3]

    flow_to_predicate_attention: Tensor
    diagnostics: dict[str, Tensor | float | list]
```

### 4.6 Semantic reason memory

Total tokens:

```text
32 predicate + 13 semantic flow + 3 lane + 2 axis + 3 direction + 1 null = 54
```

```python
@dataclass
class SemanticReasonMemory:
    values: Tensor                     # [B,54,768]
    mask: Tensor                       # [B,54] bool
    confidence: Tensor                 # [B,54]
    names: tuple[str, ...]
    type_ids: Tensor                   # predicate/flow/lane/axis/direction/null
    axis_ids: Tensor                   # longitudinal/lateral/both/none
    evidence_maps: Tensor              # [B,32,54,H,W]
    lineage: list[dict]
    semantic_state: Tensor             # [B,768], query aggregate, not mean
```

### 4.7 Formal output

```python
@dataclass
class FlowCalV2TrainOutput:
    action_text_loss: Tensor
    explanation_text_loss: Tensor
    speed_loss: Tensor
    course_loss: Tensor
    auxiliary_loss: Tensor
    total_loss: Tensor

    baseline_masked_logits: Tensor
    enhanced_masked_logits: Tensor

    control_base_prediction: Tensor    # [B,32,2]
    control_final_prediction: Tensor   # [B,32,2]
    control_hidden: Tensor             # [B,32,768]

    loss_components: dict[str, Tensor]
    gradient_diagnostics: dict[str, Tensor]
    bundle: FlowCalV2Bundle
```

---

## 5. ADAPT baseline preservation

### 5.1 Video backbone

Refactor or wrap the current ADAPT Video-Swin code, but require:

- one Video-Swin forward per batch;
- released checkpoint load report;
- native multiscale outputs;
- temporal alignment to 32 only for the reasoning branch;
- original dense tokens retained for baseline text/control;
- ADAPT `fc.weight` and `fc.bias` loaded exactly.

### 5.2 Real ADAPT motion transformer

Modify `src/modeling/load_sensor_pred_head.py` compatibly:

```python
class Sensor_Pred_Head:
    def encode(self, img_feats: Tensor) -> Tensor:
        ...

    def predict(self, img_feats: Tensor, frame_num: int) -> tuple[Tensor, Tensor]:
        # returns prediction and hidden
        ...

    def forward(self, img_feats, car_info=None, frame_num=None, return_hidden=False):
        # car_info may provide target and default length, but must never enter encoder input
        ...
```

Create `ADAPTMotionBackbone` that:

1. instantiates the same sensor head;
2. loads `sensor_pred_head.*` from the released ADAPT checkpoint;
3. calls `predict(dense_tokens_projected, steps=32)`;
4. returns exact baseline prediction and hidden;
5. computes targets/loss outside the predictor;
6. proves predictions are independent of target values when `frame_num` is fixed.

With all V2 deployment gates at zero, the text and control paths must numerically reproduce ADAPT within BF16 tolerance.

### 5.3 Caption tensorizer contract

Implement `resolve_adapt_text_contract()`:

Priority:

1. checkpoint-side `args.json`, `training_args.bin`, or resolved config if present;
2. official repository BDD-X config;
3. explicit V2 YAML defaults.

Write the source and values to:

```text
adapt_text_contract.json
run_manifest.json
```

Formal defaults:

```text
mask_prob: 0.5
max_masked_tokens: 45
max_seq_a_length: 15
max_seq_length: 30
use_sep_cap: true
```

The preflight must compare action/explanation masked-token counts against the original ADAPT dataloader on identical samples.

---

## 6. Local partial transport V2

File: `fate_x/acpr_flow_v2/local_partial_transport.py`

### 6.1 Algorithm

For each adjacent aligned time step:

1. project and L2-normalize fused-grid features;
2. estimate a common camera shift from coarse-grid correlation;
3. center each source token's candidate neighborhood at:
   ```text
   source position + estimated common shift
   ```
4. use a 5×5 local candidate set plus dustbin;
5. score cosine similarity minus spatial penalty;
6. mask out-of-frame candidates rather than wrapping with `torch.roll`;
7. row-normalize with softmax;
8. expose expected displacement and dustbin confidence.

No dense `N×N` matrix. No column normalization.

### 6.2 Warping API

Implement:

```python
def warp_source_map_to_current(
    source_map: Tensor,               # [B,P,H,W]
    transport: LocalTransportOutput,
    step: int,
) -> Tensor:                          # [B,P,H,W]
```

Use scatter-add over candidate indices. Dustbin mass is discarded. Preserve mass up to dustbin loss.

### 6.3 Required tests

- zero shift;
- known ±1/±2 translation;
- invalid border candidates;
- unmatched region to dustbin;
- row sum;
- gradient to projection;
- memory report proves no global matrix;
- warped Gaussian peak moves correctly.

---

## 7. Transported named predicate trajectories

File: `temporal_predicate_tracker.py`

Use exact 32 predicate names and order from the existing ACPR ontology.

### 7.1 Forward recurrence

At time 0:

```text
score = current visual score + log region prior
```

At time t > 0:

```text
transported_prior = warp(previous predicate attention)
beta = learned bounded beta × previous confidence × transport reliability
score = current visual score + log region prior
        + beta × log(transported_prior + eps)
attention = entmax15(score / temperature)
```

Current visual evidence is mandatory at every time step. The transported prior cannot replace it.

### 7.2 Confidence and motion

Confidence must use:

- attention concentration;
- presence confidence;
- retained transport mass;
- feature agreement.

Relative motion must use transport-weighted expected displacement and subtract common camera shift. Center-of-mass displacement may be used as an auxiliary consistency term but not the sole motion estimate.

### 7.3 Dynamic descriptor

Compute and expose:

```text
now
confidence-weighted history
least-squares temporal trend
first-difference magnitude
second-difference volatility
presence rate
relative-motion mean
relative-motion variance
```

Concatenate and project to `state_dim`.

Temporal reverse must invert the trend of monotonic synthetic sequences. Oscillation must increase volatility.

---

## 8. Predicate-conditioned lane-wise mesoscopic flow field

File: `lane_flow_field.py`

This is the primary video-specific extension beyond BDD-OIA.

### 8.1 Soft lane/corridor masks

Build geometry priors for left/center/right and refine them using:

```text
drivable_left
drivable_center
drivable_right
ego_lane_centered
lane_left_available
lane_right_available
```

The final mask is soft, normalized, and sample/time dependent. It must not use manual boxes.

### 8.2 Per-region statistics

Use vehicle/obstacle/cyclist/pedestrian predicates and confidence:

```text
occupancy
camera-compensated apparent motion x/y
motion magnitude
motion coherence
stopped tendency
queue pressure
presence trend
```

Do not claim physical density or metric speed.

### 8.3 Temporal encoder

Use a lightweight depthwise temporal convolution or two-layer gated temporal block over 32 steps. It must be order-sensitive and substantially cheaper than a full spatiotemporal transformer.

Outputs:

```text
temporal lane tokens [B,32,3,D]
lane descriptors [B,3,D]
```

### 8.4 Required behavior

Synthetic tests:

- center occupancy increasing;
- left lane moving more freely than center;
- right blockage;
- queue forming;
- queue releasing.

Each must produce distinct descriptors and correct lane-advantage signs.

---

## 9. Axis-aware flow composer

File: `axis_aware_flow_composer.py`

### 9.1 Original 13 semantic factors remain

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

Each token combines:

```text
predicate evidence
lane-flow evidence
learned semantic-name embedding
grammar support/contradiction prior
```

### 9.2 Axis and direction attributes

Output:

```text
axis: longitudinal / lateral
direction: left / neutral / right
```

Direction must depend on signed left/right differences and directional predicates:

```text
open_left_gap vs open_right_gap
vehicle_left vs vehicle_right
merging_left_context vs merging_right_context
left_turn_region vs right_turn_region
left-lane vs right-lane occupancy/motion advantage
```

A directionless `merge_lane_constraint` token cannot be the sole course input.

### 9.3 Weak control-derived supervision

Use original train controls only as detached targets, never as model inputs.

Derive:

```text
normalized speed delta
shortest circular course delta
lateral relevance mask
left / neutral / right target
```

Use low-weight axis/direction losses. This is intermediate-state supervision from original labels, not new annotation.

Audit must prove replacing `car_info` values while holding the input video fixed does not change forward predictions before loss computation.

---

## 10. Contextual free-text reason target

File: `contextual_reason_target.py`

### 10.1 Frozen contextual encoder

Load a separate frozen BERT encoder from the local `bert_dir` in eval mode. It must:

- process raw action and justification separately;
- use contextual hidden states, not static token embedding means;
- mean-pool non-special valid tokens;
- use no gradient;
- save no embedding cache;
- run only during training/diagnostic target construction.

### 10.2 Action-subspace tracker

Maintain a train-only, detached covariance of normalized action embeddings. At epoch boundaries compute a rank-16 orthonormal action basis. Save only the basis and statistics in checkpoints.

No test text may update the basis.

### 10.3 Target

```text
a = normalized contextual action embedding
e = normalized contextual explanation embedding
pair projection = (e·a)a
global projection = BᵀBe
r* = normalize(e - rho_pair*pair_projection - rho_global*global_projection)
```

Defaults:

```text
rho_pair = 0.50
rho_global = 0.50
rank = 16
```

If the basis is not ready in the first epoch, use only the pair projection.

Log:

```text
cos(reason, action) before/after
target norm
reason length
basis rank
```

Inference cannot instantiate or call the target encoder.

---

## 11. Free-text positive–unlabeled supervision V2

File: `pu_targets.py`

### 11.1 Rule loading

Load text rules from YAML, not a hidden hardcoded list. Preserve positive, contradiction, unknown, and reliability masks for predicates and semantic flow factors.

### 11.2 Unknown schedule

```text
semantic recovery epochs 0–2: unknown weight = 0
later stages: group-normalized unknown regularizer <= 0.005
```

Do not apply a fixed negative BCE independently to every unknown predicate.

Recommended unknown regularizer:

```text
lambda_unknown × mean(sigmoid(logits_unknown))
```

normalized by unknown count per sample and group.

### 11.3 Audit

For each formal epoch log:

```text
known positive count
known contradiction count
unknown count
positive loss
contradiction loss
unknown regularizer
```

---

## 12. Semantic reason memory V2

File: `semantic_reason_memory.py`

### 12.1 Token construction

```text
32 predicate tokens
13 semantic flow tokens
3 lane tokens
2 axis tokens
3 direction tokens
1 null token
```

Every token includes:

- projected state;
- learned semantic-name embedding initialized from frozen BERT name encoding;
- confidence gate;
- evidence map;
- type and axis metadata;
- support lineage.

### 12.2 No equal averaging

Compute `semantic_state` with a learned null-safe query over memory, weighted by confidence and mask. Store the attention distribution.

### 12.3 Axis views

Expose masks:

```text
longitudinal memory:
  front/traffic-light/clear/crowded/queue/following/center lane/longitudinal axis

lateral memory:
  left/right lane/vehicle/gap/merge/turn/lane descriptors/lateral axis/direction
```

Action/explanation may read all memory. Speed must primarily read longitudinal memory. Course must primarily read lateral memory.

Audit must show speed attention mass is higher on longitudinal tokens and course attention mass is higher on lateral tokens on targeted synthetic cases.

---

## 13. Semantic Gradient Firewall

File: `semantic_gradient_firewall.py`

This is the core fix for explanation degradation.

### 13.1 Consumer-specific gradient scaling

Forward values remain identical; only gradients are scaled.

```python
def scaled_gradient(x: Tensor, scale: float) -> Tensor:
    return x.detach() + scale * (x - x.detach())
```

Defaults:

```text
explanation = 1.00
action      = 0.20
speed       = 0.075
course      = 0.075
```

Use explicit consumer views:

```text
memory_for_explanation
memory_for_action
memory_for_speed
memory_for_course
```

### 13.2 Efficient representation-level conflict projection

Do not run full parameter-level PCGrad over the whole network.

For semantic loss and control loss, calculate gradients with respect to the shared reason-memory tensor:

```python
g_sem = grad(L_explanation + λ_reason*L_reason, reason_memory)
g_ctrl = grad(L_speed + L_course + λ_future*L_future, reason_memory)
```

When their dot product is negative:

```text
g_ctrl_projected =
g_ctrl - (dot(g_ctrl,g_sem) / ||g_sem||²) * g_sem
```

Add a detached surrogate correction:

```text
L_firewall =
<reason_memory, stopgrad(g_ctrl_projected - g_ctrl)>
```

Therefore the final upstream gradient is:

```text
g_sem + g_ctrl_projected + other gradients
```

The forward loss value remains unchanged apart from a zero-valued surrogate.

Log per optimizer step:

```text
pre_projection_cosine
post_projection_cosine
conflict_rate
correction_norm
semantic_grad_norm
control_grad_norm
```

Skip safely when either gradient norm is zero.

### 13.3 Scope

The correction acts at reason-memory representation level and therefore protects:

```text
predicate/flow-to-reason projections
semantic memory
SECA memory projections
```

It does not modify the ADAPT video/text baseline outside the reason path.

---

## 14. Temporal SECA V2

File: `temporal_seca.py`

Hook location remains after BERT multimodal hidden states and before the LM prediction head.

### 14.1 Separate readers

Implement separate action and explanation query projections, gates, and gradient views.

```text
action tokens       → memory_for_action
explanation tokens  → memory_for_explanation
image hidden        → unchanged
```

Use entmax15 over all valid reason tokens and include the null token.

Defaults:

```text
action gate max = 0.08
explanation gate max = 0.25
output projection = Xavier
gate raw init = 0
```

### 14.2 ADAPT equivalence

At zero gate:

```text
teacher-forced logits equal ADAPT
generated action equal ADAPT
generated explanation equal ADAPT
```

The module must still execute; equivalence cannot be obtained by bypassing its construction.

### 14.3 Segment correctness

Test both teacher forcing and `use_sep_cap` generation. Verify generated action and explanation invoke the correct reader. Do not infer segment solely from a fragile fixed token position.

---

## 15. Axis-aware control mediation

Files:

```text
adapt_motion_backbone.py
axis_aware_control_adapter.py
```

### 15.1 Base

```text
dense ADAPT video tokens
→ released ADAPT sensor motion transformer
→ base prediction [B,32,2]
→ hidden [B,32,768]
```

### 15.2 Residual readers

```text
speed query  → longitudinal memory
course query → lateral memory
```

Separate Q/K/V, entmax attention, delta heads, and gates.

Apply train-split control scale:

```text
speed_final = speed_base + gate_speed * sigma_speed * tanh(delta_speed)
course_final = circular_wrap(
    course_base + gate_course * sigma_course * tanh(delta_course)
)
```

Maximum residual:

```text
0.15 train standard deviations
```

Gate raw initialization is zero.

No flow logits, factor probabilities, or lane statistics may directly enter a control MLP. They must be represented as reason-memory tokens.

### 15.3 No target leakage

`car_info` is used only:

- to choose target length when an explicit length is absent;
- to calculate losses and weak detached labels.

Changing target values while keeping length/video fixed must not change predictions.

---

## 16. True prefix-to-future auxiliary task

File: `prefix_future.py`

Use 25% of eligible train batches.

1. Run the Video-Swin backbone once for all 32 frames.
2. Slice precomputed aligned reasoning grids to first 24 frames.
3. Recompute transport, predicate, lane flow, flow states, and reason memory from the prefix only.
4. Predict the final 8 speed/course steps.
5. The final 8 frame features and control targets cannot enter prefix state construction.

Do not use the full-video pooled reason state as a prefix state.

Loss weight: `0.01`.

---

## 17. Contradiction-aware Temporal HardPair

File: `temporal_hardpair.py`

Activate only in the conflict-aware joint stage.

A negative pair must satisfy:

```text
high action similarity
low reason semantic similarity
AND explicit predicate/flow contradiction score above threshold
```

Text embedding distance alone is insufficient.

Defaults:

```text
queue size = 4096
max pairs per batch = 32
start epoch = 8
weighted gradient budget <= 5% of semantic reason gradient
```

Log why pairs are eligible/ineligible.

---

## 18. Losses

File: `acpr_flowcal_v2_losses.py`

### 18.1 Control normalization

Stream train control labels once without loading images and save:

```text
control_train_stats.json
```

Compute:

```text
mean/std per signal
valid count
circular course handling
```

Use normalized SmoothL1/Huber:

```text
speed loss  = Huber((pred-target)/sigma_speed)
course loss = Huber(shortest_circular_delta(pred,target)/sigma_course)
```

### 18.2 Main losses

Base weights in joint stage:

```text
action_text                 1.00
explanation_text            1.00
speed_normalized            0.03
course_normalized           0.03
predicate_PU                0.03
flow_PU                     0.03
reason_semantic             0.08
transport_consistency       0.01
lane_temporal_consistency   0.01
axis_direction_weak         0.01
prefix_future               0.01
hardpair                    dynamically capped
action_delta_KL             0.08
explanation_delta_KL        0.03
parameter_anchor            0.0001
memory_diversity            0.0005
firewall_surrogate          value zero, gradient only
```

### 18.3 Preserve terms

- `action_delta_KL`: baseline vs enhanced teacher-forced action logits.
- `explanation_delta_KL`: lower-weight baseline vs enhanced explanation logits.
- `parameter_anchor`: trainable BERT/Swin/sensor parameters vs the loaded ADAPT initialization.

The base and enhanced logits must come from identical current baseline weights, while parameter anchoring prevents both from drifting far from ADAPT.

### 18.4 Required logging

Log raw, weighted, and gradient norm for every nonzero component. Reject any configured loss that is constant zero or omitted from total.

---

## 19. Explanation sequence-level tuning

File: `fate_x/losses/explanation_scst.py`

Implement real self-critical sequence training for the final two epochs.

### 19.1 Decoder API

Add or wrap:

```python
generate_explanation_with_logprobs(
    ...
) -> GeneratedSequence:
    token_ids
    token_logprobs
    decoded_text
```

Generate:

- one sampled explanation;
- one greedy baseline explanation under `torch.no_grad()`.

### 19.2 Reward

```text
reward = 0.70 * CIDEr_exp
       + 0.30 * METEOR_exp
       - hallucination_penalty
```

Hallucination penalty triggers when generated explanation contradicts active high-confidence predicate/flow states.

### 19.3 Mixed objective

```text
L = 0.50 * explanation_CE
  + 0.10 * SCST
  + 0.10 * action_preserve
```

Only 25% of batches need the expensive SCST path. Other batches use CE/preserve.

During SCST stage freeze:

```text
Video Swin
predicate tracker
transport
lane flow
flow composer
reason memory
ADAPT motion head
control adapters
```

Train only:

```text
explanation SECA query/output/gate
BERT final layer
LM head
```

Action and control outputs must remain protected.

---

## 20. Stage controller and exact 15-epoch schedule

File: `train_acpr_flowcal_v2.py`

No metric-based early stop. Training continues through all stages unless:

- all configured work completes;
- the user explicitly requests stop;
- unrecoverable hardware/OS loss occurs.

### Stage R — Semantic Recovery

```text
epochs 0–2, 3 epochs
```

Train:

```text
transport projection
predicate tracker
lane flow field
flow composer
reason memory
explanation SECA
action SECA gate/query at protected scale
```

Freeze:

```text
Video Swin
ADAPT fc
BERT body and LM head
ADAPT motion head
control adapter gates/deltas
```

Settings:

```text
control losses = 0
HardPair off
unknown PU weight = 0
firewall logs semantic path only
```

### Stage M — Axis-Aware Motion

```text
epochs 3–7, 5 epochs
```

Train additionally:

```text
axis/direction heads
speed/course reason readers
control adapter gates/deltas
prefix-future head
```

Keep ADAPT motion head frozen.

Enable normalized control, weak axis/direction, and prefix-future losses.

### Stage J — Conflict-Aware Joint

```text
epochs 8–12, 5 epochs
```

Train:

```text
all V2 modules
BERT layer 11
Video-Swin final stage
ADAPT motion decoder and final motion encoder layer
```

Enable:

```text
semantic gradient firewall
contradiction-aware HardPair
small unknown regularizer
parameter anchoring
```

### Stage S — Explanation SCST

```text
epochs 13–14, 2 epochs
```

Freeze all visual/control/state modules. Train only explanation language adapter and final language layer/head.

### Calibration

After epoch 14:

```text
one deterministic train-calib pass
no model-weight update
fit Sequence-CalAlign
then run one final full test evaluation
```

---

## 21. Optimizer and scheduler

### 21.1 Learning rates

```text
transport                    5e-5
predicate tracker            5e-5
lane flow                    1e-4
flow composer                5e-5
reason memory                5e-5
SECA                          2e-5
control adapters             2e-5
prefix future                2e-5
HardPair projection          1e-5
BERT layer 11                5e-6
Video-Swin final stage       2e-6
ADAPT motion final layer     2e-6
ADAPT motion decoder         5e-6
LM head during SCST          2e-6
```

Weight decay:

```text
new weights 0.01
backbone weights 0.05
bias/norm/gates 0
```

### 21.2 Stage-aware scheduler

Implement a real stage-aware scheduler:

- 5% warmup inside each stage;
- cosine decay to 0.10 of stage initial LR;
- scheduler state saved/restored;
- frozen groups receive multiplier 0;
- every optimizer step calls scheduler exactly once.

Do not rebuild optimizer silently and lose moments. If rebuilding at a stage boundary is used, transfer state by parameter identity and write a state-transfer report.

### 21.3 Gradient clipping

Global clip norm `1.0`. Also log pre/post clip norm.

---

## 22. Checkpoint migration and initialization

Implement `CheckpointMigratorV1ToV2`.

Preferred initialization order:

1. historical ACPR checkpoint associated with the strongest early text epoch;
2. another valid complete V1 checkpoint chosen explicitly by path;
3. released ADAPT checkpoint if no compatible V1 checkpoint exists.

Never use `.tmp`.

Migration:

- exact load of Video-Swin, ADAPT fc, captioning BERT/LM, sensor head;
- load compatible predicate query/projection weights;
- map original 13 flow factor weights to V2 semantic factors;
- map 32 predicate and 13 flow semantic memory embeddings;
- initialize lane/axis/direction/firewall/course-specific modules newly;
- optionally initialize speed reader from V1 control adapter;
- write exact loaded/missing/unexpected key report.

Allowed missing keys must be an explicit V2-only prefix allowlist. Any baseline key mismatch blocks training.

---

## 23. Sequence-CalAlign V2

File: `sequence_calalign.py`

Use deterministic 10% `train_calib` IDs split from train by hash. No test labels in fitting.

Fit independently:

```text
alpha_action
alpha_explanation
alpha_speed
alpha_course
temperature_action
temperature_explanation
```

Every alpha grid includes zero.

Apply:

```text
text logits = base + alpha*(enhanced-base)
control = base + alpha*(enhanced-base)
```

For course, interpolate via shortest circular residual.

Save scales in:

```text
sequence_calalign.json
checkpoint metadata
final run manifest
```

Test selects checkpoints, but test must not fit calibration parameters.

---

## 24. Evaluation and best-checkpoint selection

Every epoch:

1. finish train epoch;
2. atomically save `checkpoint_latest.pth`;
3. run smoke test evaluation;
4. run full **test-only** caption and continuous-control evaluation;
5. run lightweight traffic-state audit;
6. save best checkpoints;
7. continue to next fixed epoch.

No validation DataLoader.

### 24.1 Text metrics

Separate:

```text
description/action:
BLEU1-4, METEOR, ROUGE-L, CIDEr, SPICE

explanation:
BLEU1-4, METEOR, ROUGE-L, CIDEr, SPICE
```

Use the same tokenization and evaluation scripts as ADAPT.

### 24.2 Control metrics

Per speed/course:

```text
RMSE
MAE
Acc@0.1
Acc@0.5
Acc@1
Acc@5
Acc@10
```

Course errors use shortest circular delta.

### 24.3 Checkpoints

```text
checkpoint_latest.pth
checkpoint_best_text.pth
checkpoint_best_explanation.pth
checkpoint_best_control.pth
checkpoint_best_joint.pth
checkpoint_best_test.pth
```

`checkpoint_best_test.pth` uses an explanation-first, control-safe lexicographic test selector:

1. eligible if speed and course RMSE are each no worse than 1.02× the measured ADAPT baseline;
2. among eligible candidates maximize:
   ```text
   CIDEr_exp,
   CIDEr_des + CIDEr_exp,
   METEOR_exp,
   negative normalized control RMSE
   ```
3. if no candidate is eligible, minimize total control violation first, then maximize text.

Write the full comparison tuple; do not hide it in one arbitrary scalar.

Protocol tag:

```text
test_selected_user_requested
```

Beam size formal: `1`, unless an exact ADAPT-equivalent multi-beam sep-cap implementation is dynamically proven.

---

## 25. Traffic-flow relevance and faithfulness audit

After training, use `checkpoint_best_test.pth` and calibrated deployment scales.

### 25.1 Per-sample middle output

Write JSONL:

```text
top named predicates over time
left/center/right lane flow descriptors
regime/phase/source probabilities
axis and direction probabilities
reason-memory attention for action/explanation/speed/course
base/enhanced action and explanation
base/enhanced speed/course
```

### 25.2 Interventions

No retraining:

```text
all flow off
longitudinal flow off
lateral flow off
individual top flow factor off
individual predicate off
evidence tube off
random equal-mass evidence off, 5 seeds
temporal shuffle
temporal reverse
last-frame-only
prefix 8/16/24/32
```

Each intervention must rerun all affected downstream computations.

### 25.3 Normalized effects

```text
speed effect = ||speed - speed_cf|| / sigma_speed
course effect = circular ||course - course_cf|| / sigma_course
text effect = token KL + generated-text metric delta
```

Report:

```text
all samples
longitudinal subset
left/right turn subset
merge/lane-change subset
traffic-signal subset
queue/following subset
```

Do not require all traffic states to affect course. The expected claim is:

```text
traffic flow has broad longitudinal influence
and conditional lateral influence in lateral-relevant scenes
```

### 25.4 Direction consistency

For lateral-relevant samples, test whether left/right state interventions change course in the expected signed direction.

### 25.5 Statistics

Use paired bootstrap confidence intervals and fixed-seed paired permutation tests. Describe results as model-internal counterfactual dependence, not real-world causal effect.

---

## 26. Visualization

File: `acpr_flowcal_v2_renderer.py`

For each selected sample produce a high-resolution PNG and source JSON containing:

1. 8 keyframes with top predicate evidence overlays;
2. 32-step predicate probability trajectories;
3. left/center/right occupancy, motion, and queue-pressure ribbons;
4. regime/phase/source/axis/direction curves;
5. hierarchical support graph:
   ```text
   visual predicate → lane/flow state → reason token → output
   ```
6. action/explanation token-to-reason attention;
7. base/enhanced/flow-off longitudinal speed plot;
8. base/enhanced/lateral-off course plot;
9. generated baseline/enhanced/counterfactual text;
10. evidence deletion vs equal-mass random effect;
11. sample ID, checkpoint SHA, config SHA, and tensor lineage.

No manual boxes. No fabricated effects. No template-only counterfactual sentence.

Dataset atlas must:

- group by dominant flow state and action;
- show top prototypes and failure cases;
- include intervention effect distributions;
- link every row to source JSON/PNG;
- generate standalone HTML plus JSON index.

---

## 27. Formal experiment suite

Use one primary V2 training run rather than many blind ablations.

### Run 0 — Released ADAPT baseline

No training. Full test text/control evaluation and save measured baseline.

### Run 1 — Current V1 references

Evaluate:

- historical early text-best complete checkpoint;
- strict/current complete checkpoint.

Do not mix their claims.

### Run 2 — Full V2

Fixed 15 epochs:

```text
3 semantic recovery
5 axis-aware motion
5 conflict-aware joint
2 explanation SCST
```

Then Sequence-CalAlign and final test.

### Run 3 — No-retrain mechanism suite

All interventions and temporal tests from section 25.

### Run 4 — Visual/atlas export

Use best-test checkpoint.

No additional full retraining ablation is mandatory before these complete. Stage checkpoints and no-retrain switches provide the first analysis. Additional controlled retraining is allowed only after the main suite and only with an explicit question.

---

## 28. GPU and direct-image configuration

Hardware target: single 48 GiB GPU.

Use:

```text
direct frames [B,32,3,224,224]
BF16
gradient checkpointing
no image/video feature cache
no token cache
```

### 28.1 Main-stage memory probe

Try in order:

```text
micro batch 6, accumulation 6, effective 36
micro batch 5, accumulation 7, effective 35
micro batch 4, accumulation 8, effective 32
micro batch 3, accumulation 11, effective 33
micro batch 2, accumulation 16, effective 32
```

For each:

- 3 warmup + 30 measured full forward/backward steps;
- formal losses;
- no OOM;
- no NaN/Inf;
- no skipped optimizer step;
- peak reserved <= 44.5 GiB.

Choose the largest stable micro-batch. Preferred measured range is 30–42 GiB, but do not allocate dummy memory.

### 28.2 SCST probe

Separate candidates:

```text
batch 3 / accum 11
batch 2 / accum 16
batch 1 / accum 32
```

### 28.3 Precision

Prefer BF16 with no GradScaler. FP16 fallback is allowed only if BF16 unsupported and must use a conservative scaler, overflow counter, and finite guard.

---

## 29. Tests required before review

Create focused tests under `tests/acpr_flowcal_v2/`.

### Static/contracts

```text
test_v2_formal_import_graph.py
test_v2_config_binding.py
test_v2_named_outputs.py
test_v2_no_cache_contract.py
test_v2_adapt_text_contract.py
```

### Baselines

```text
test_v2_adapt_video_load.py
test_v2_adapt_motion_equivalence.py
test_v2_motion_target_independence.py
test_v2_zero_gate_fallback.py
```

### Transport/predicates

```text
test_v2_local_transport.py
test_v2_transport_warp.py
test_v2_transported_predicate_tracker.py
test_v2_dynamic_descriptor.py
test_v2_camera_compensated_motion.py
```

### Lane/flow

```text
test_v2_lane_flow_field.py
test_v2_axis_direction_flow.py
test_v2_flow_grammar.py
test_v2_lateral_direction_sign.py
```

### Reason/text

```text
test_v2_contextual_reason_target.py
test_v2_action_subspace_train_only.py
test_v2_pu_unknown_schedule.py
test_v2_reason_memory_54.py
test_v2_seca_segment_readers.py
```

### Gradient firewall

```text
test_v2_scaled_gradient.py
test_v2_representation_pcgrad.py
test_v2_firewall_forward_invariance.py
test_v2_firewall_gradient_ratios.py
```

### Control/loss

```text
test_v2_axis_control_adapter.py
test_v2_course_circular_residual.py
test_v2_control_normalization.py
test_v2_prefix_future_no_leak.py
test_v2_hardpair_contradiction.py
```

### Trainer

```text
test_v2_stage_controller.py
test_v2_optimizer_groups.py
test_v2_scheduler_step_resume.py
test_v2_checkpoint_migration.py
test_v2_test_only_protocol.py
test_v2_best_selector.py
test_v2_scst_logprob_reward.py
```

### Intervention/visual

```text
test_v2_intervention_recompute.py
test_v2_equal_mass_random.py
test_v2_temporal_necessity.py
test_v2_renderer_schema.py
test_v2_atlas_schema.py
```

### Supervisor/audit

```text
test_v2_review_pass_binding.py
test_v2_foreground_supervisor.py
test_v2_recovery_reaudit_contract.py
```

---

## 30. Preflight gates

Formal audit must dynamically run:

### Gate A — Compile, imports, and full tests

```text
py_compile changed package files
pytest tests/acpr_flowcal_v2 -q
selected existing ADAPT regression tests
git diff --check
```

### Gate B — ADAPT equivalence

Identical real test samples with all V2 gates zero:

```text
text teacher-forced max diff <= 1e-4 BF16
generated action/explanation exact token match
control max diff <= 1e-4 BF16
```

### Gate C — 8-step real direct-image smoke

Must show:

```text
frames [B,32,3,224,224]
one Video-Swin forward
all formal losses finite
forward/backward
optimizer/scheduler step
test smoke eval
atomic checkpoint
no cache
```

### Gate D — Gradient chain

After appropriate gate warmup:

```text
transport projection
predicate queries
lane temporal encoder
flow queries
reason memory
explanation SECA
action SECA
speed reader
course reader
control gates
contextual target has no gradient
```

### Gate E — Stage execution

Run synthetic mini epochs and prove:

- exact trainables per stage;
- HardPair off/on at correct epoch;
- unknown schedule changes;
- SCST freeze list;
- scheduler multipliers and state resume.

### Gate F — 128-sample mechanism fit

Overfit a deterministic 128-sample train subset for a bounded number of updates. Require intended losses to improve without:

```text
predicate collapse
flow factor constant outputs
reason attention all-null
speed-only memory domination
course attention ignoring lateral tokens
```

This is mechanism validation, not performance reporting.

### Gate G — Temporal necessity

On real samples:

```text
shuffle/reverse changes phase/trend
last-frame-only differs from full video
prefix future is worse when temporal order is corrupted
```

### Gate H — Real intervention

Evidence-tube deletion must beat equal-mass random deletion on average for at least one intended downstream branch, and flow-off must change actual model outputs.

### Gate I — Memory

Select and record the stable batch configuration.

### Gate J — Visualization

Render one real full canvas and mini atlas. Verify every plotted value traces to a tensor or generated output.

Only then may the audit write the review pass.

---

## 31. Audit authorization artifact

Required directory:

```text
.background_runs/acpr_flowcal_v2_preflight/
```

Required pass:

```text
REVIEW_PASS_ACPR_FLOWCAL_V2.txt
```

It must bind:

```text
worktree
branch
clean local Git SHA
GitHub remote SHA
plan/config/skill hashes
test command results
baseline equivalence
direct-image/no-cache proof
stage execution
gradient proof
firewall proof
control baseline proof
temporal proof
intervention proof
visual proof
memory selection
foreground-supervisor proof
```

Any source/config change invalidates the pass. Delete the old pass, rerun all relevant gates, commit/push, and issue a new pass.

---

## 32. Foreground supervisor

`supervise_acpr_flowcal_v2_foreground.py` must be an attached process that streams child output.

Forbidden:

```text
Start-Process
Start-Job
schtasks
nohup
shell &
DETACHED_PROCESS
hidden window flags
daemon threads that outlive parent
```

Required:

- verify pass file SHA and GitHub SHA before launch;
- stream stdout/stderr live;
- heartbeat every 60 seconds even during long evaluation;
- update `supervisor_live_status.json`;
- verify epoch artifacts;
- retry transient I/O failures;
- OOM fallback to next audited micro-batch;
- atomically resume `checkpoint_latest.pth`;
- reject `.tmp`;
- never stop because metrics decline;
- accept explicit stop sentinel:
  ```text
  control/STOP_REQUESTED_BY_USER
  ```
- on recoverable code failure:
  1. preserve logs/checkpoint;
  2. debug systematically;
  3. add regression test;
  4. commit and push fix;
  5. invalidate old review pass;
  6. rerun full audit;
  7. resume latest;
  8. continue foreground supervision.

Completion is allowed only after the complete configured suite, final intervention/visual export, and final record update have finished, or the user explicitly requests stop.

---

## 33. PowerShell launch

The final launcher must directly invoke the foreground supervisor in the current console.

Conceptual command:

```powershell
cd E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_X_acpr_flowcal_v2_foreground.ps1 `
  -Config configs\acpr_flowcal_v2_bddx_32f_224.yaml `
  -OutputDir E:\sbw\FATE_Drive\active_runs\acpr_flowcal_v2_<timestamp> `
  -Device cuda `
  -RequireReviewPass `
  -ReviewPassDir .background_runs\acpr_flowcal_v2_preflight
```

The launcher must not create a hidden/background job.

---

## 34. Codex execution roles with Superpowers

### Agent A — Implementer

Use installed Superpowers skills, at minimum:

```text
writing-plans
test-driven-development
executing-plans
systematic-debugging
verification-before-completion
```

Agent A must maintain a task checklist mapping every section of this contract to code/tests/evidence.

### Agent B — Independent adversarial reviewer

After Agent A says implementation is complete:

1. start from the plan and audit skill, not Agent A's summary;
2. inspect the formal import graph;
3. run the audit independently;
4. deliberately search for YAML-only features, dead gradients, bypasses, target leakage, placeholder outputs, and fake evidence;
5. reject until every blocker has code and dynamic proof.

Use `requesting-code-review` / review skills when available.

### Review loop

```text
Agent A implement
→ Agent B reject with exact blockers
→ Agent A add test and fix
→ commit/push
→ Agent B rerun
```

No formal training before independent pass.

---

## 35. Definition of done

The task is complete only when all are true:

- V2 formal path exists and is independent of V1 formal model;
- ADAPT text and motion baselines load and zero-gate reproduce;
- direct 32-frame images are used with no feature cache;
- transported predicate trajectories are real;
- lane-wise flow field is order-sensitive;
- axis/direction state supports course;
- contextual reason target is frozen, contextual, and train-only;
- unknown text labels are not hard negatives;
- reason memory contains 54 typed tokens and lineage;
- explanation/action/speed/course use intended gradient scales;
- representation conflict projection is dynamically proven;
- speed reads longitudinal memory;
- course reads lateral memory;
- control uses real ADAPT motion hidden;
- stage schedule executes;
- scheduler executes and resumes;
- SCST uses real sampled-token log probabilities;
- per-epoch full test evaluation and test-best selection work;
- traffic-flow intermediate outputs are saved;
- flow/predicate/evidence/temporal interventions change real outputs;
- hierarchical canvas and atlas are non-placeholder;
- memory probe selects a stable 30–42 GiB preferred configuration when feasible;
- audit pass binds clean pushed `flowtrace_pmt_v1` SHA;
- foreground supervisor runs the complete suite without detached execution;
- code, tests, plan, skill, and config are pushed to GitHub;
- the established task/findings/progress records contain exact commands, SHAs, metrics, failures, fixes, and remaining boundaries.

Anything less is implementation-in-progress, not authorization to train.
