# ACPR-FlowCal++ V1
# Codex Code-Level Implementation, Review, Experiment, and Foreground-Supervision Plan

**Repository:** `https://github.com/d2116056543-lab/FATE-X`  
**Existing branch:** `flowtrace_pmt_v1`  
**Existing worktree:** `E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree`  
**No new worktree and no new branch. Modify the current worktree and push the same branch.**  
**Formal method name:** `ACPR-FlowCal++ V1`  
**Primary config:** `configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml`  
**Formal execution:** direct 32-frame image tensors; no learned image/video feature cache; test-only epoch evaluation; best selected on test as explicitly requested.

---

## 0. Objective and non-negotiable semantic chain

The target is not a generic FlowTrace patch. It is the video/free-text extension of the successful BDD-OIA `ACPR-CalAlign V1.2` method.

The complete model chain must be:

```text
32-frame monocular driving clip
  -> dual-scale Video Swin feature grids
  -> 32 named ACPR local predicate trajectories
  -> factorized mesoscopic traffic-flow states
  -> high-dimensional predicate/flow-enriched reason memory
  -> sparse action-specific / explanation-specific / control-specific reason reading
  -> original BDD-X action narration, original BDD-X justification, speed/course
  -> Sequence-CalAlign deployment scaling
```

The central invariant is:

```text
local predicate / traffic-flow state
    must influence action and control only through reason memory
```

Forbidden shortcuts:

```text
flow -> final action logits directly
flow -> final control directly
predicate -> final token logits directly
generated explanation text -> action input at inference
GT action/justification -> inference state
post-hoc candidate selection
latent-state norm falsely reported as intervention faithfulness
attention-only visualization presented as causal evidence
```

Required directed path:

\[
V_{1:T}
\rightarrow P_{1:T}
\rightarrow F
\rightarrow R
\rightarrow \{A,E,U\}
\]

where:

- `P`: 32 named local ACPR predicates;
- `F`: factorized traffic-flow regime, phase, and source;
- `R`: high-dimensional reason memory;
- `A`: action narration;
- `E`: original BDD-X justification;
- `U`: speed/course control sequence.

When all new deployment scales are zero, output must exactly recover the released ADAPT baseline.

---

## 1. Mandatory pre-command context and operating rules

Before changing code, running tests, training, evaluating, pushing, or managing processes, Codex must read:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md
```

All durable training/experiment status must be appended only to those three files.

The implementation plan and audit skill are non-status runbooks and may be committed under:

```text
docs/runbooks/
.codex/skills/
```

Do not create extra per-run status Markdown files.

Do not commit:

```text
.background_runs/
datasets/
frame TSV data
processed_video_info/
checkpoints
*.pt
*.pth
generated logits
generated videos
generated PNG batches
memory probes
```

---

## 2. Current-worktree safety procedure

The user explicitly requires direct modification of the current worktree.

Run:

```powershell
$Repo = "E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree"
Set-Location $Repo

git branch --show-current
git remote -v
git status --porcelain
git fetch github
git rev-parse HEAD
git ls-remote github refs/heads/flowtrace_pmt_v1
```

Required branch:

```text
flowtrace_pmt_v1
```

### 2.1 Do not reset or discard existing local changes

If the worktree is dirty:

1. inspect every modified/untracked code file;
2. save a patch under ignored storage:
   ```powershell
   New-Item -ItemType Directory -Force .background_runs\pre_acpr_flowcalpp_snapshot | Out-Null
   git diff > .background_runs\pre_acpr_flowcalpp_snapshot\working_tree.patch
   git status --porcelain > .background_runs\pre_acpr_flowcalpp_snapshot\status.txt
   ```
3. separate run artifacts from code;
4. run `py_compile` and current targeted tests;
5. commit intended existing code as a safety snapshot on the same branch:
   ```text
   Snapshot pre-ACPR-FlowCal++ FlowTrace V1 state
   ```
6. push the snapshot to `github/flowtrace_pmt_v1`;
7. verify local and remote SHA equality.

Do not use `git reset --hard`. Do not overwrite local Codex fixes blindly.

### 2.2 Branch synchronization rule

Before formal training:

```text
local HEAD == github/flowtrace_pmt_v1 HEAD
git status --porcelain == empty
```

Any code/config change after review invalidates the review pass and requires:

```text
test -> commit -> push -> audit again
```

---

## 3. Superpowers and two-role workflow

Codex must discover and use the available equivalents of:

```text
using-superpowers
writing-plans
test-driven-development
systematic-debugging
verification-before-completion
executing-plans
requesting-code-review / receiving-code-review
```

Use two explicit roles.

### Agent A — implementer

Agent A must:

- inspect actual current import and decode paths;
- write a file-by-file implementation manifest;
- implement the formal architecture;
- write tests with each component;
- expose tensor contracts and gradient contracts;
- never fabricate intervention or visualization values;
- never start formal training.

### Agent B — adversarial reviewer

Agent B must:

- review the plan before implementation;
- inspect actual diffs and actual call graph;
- run static and dynamic checks;
- reject dead modules, unused config, detached paths, fake artifacts, and placeholder renderers;
- verify exact ADAPT fallback;
- authorize training only after all gates pass.

Required cycle:

```text
plan draft
-> Agent B review
-> correction
-> implementation
-> unit tests
-> integrated direct-image smoke
-> 128-sample mechanism overfit
-> strict audit
-> commit and push
-> strict audit on exact clean pushed HEAD
-> REVIEW_PASS
-> foreground experiment suite
```

---

## 4. Current V1 defects that formal code must eliminate

The current branch may retain old files for history, but the formal path must not import or execute them.

Formal code must eliminate these known V1 defects:

1. **Positional tuple parsing**
   - no `outputs[-2]`, `outputs[-3]` protocol;
   - model/trainer communication must be a typed named output.

2. **Disconnected reason supervision**
   - reason targets and PU targets must enter the real loss;
   - reason head gradients must be nonzero in a real image batch.

3. **Triple-product PMT cold-start**
   - do not use `token * state * reason` followed by a zero output matrix;
   - use Temporal SECA with Xavier output projection and a zero ReZero scalar gate.

4. **Global Sinkhorn one-to-one assumption**
   - replace with local row-normalized partial transport and a per-source dustbin.

5. **Mean-only traffic state**
   - preserve `now`, `history`, `trend`, and `volatility`.

6. **Invalid interventions**
   - deleting a display map is not an intervention;
   - interventions must recompute every downstream component.

7. **Placeholder visualization**
   - one grayscale state map is not a FlowTrace/ACPR visualization;
   - full Canvas and dataset Atlas are mandatory.

8. **Configuration not wired to optimizer**
   - every LR/freeze/schedule field must control real parameter groups.

9. **`learn_mask_enabled` conflict**
   - the formal ACPR-FlowCal++ path must not enable the legacy learned sparse mask.

10. **FP16 overflow**
    - formal training prefers BF16 after runtime support check.

Legacy modules may remain under the repository, but formal imports and the formal resolved config must show:

```text
legacy_flowtrace_v1_enabled = false
legacy_token_pmt_enabled = false
legacy_sinkhorn_transport_enabled = false
learn_mask_enabled = false
```

---

## 5. Formal architecture and tensor contracts

### 5.1 Input and shared ADAPT backbone

Input:

```python
frames: [B, 32, 3, 224, 224]
car_info: [B, 2, 32]  # course, speed; invalid rows may contain -1
```

Use one Video Swin forward.

Required outputs:

```python
fine_stage:   [B, C3, T', Hf, Wf]  # expected Hf=Wf=14
coarse_stage: [B, C4, T', Hc, Wc]  # expected Hc=Wc=7
dense_tokens: [B, T' * Hc * Wc, D_adapt]
```

All dimensions must be inferred from tensors.

Fuse:

\[
X_t = \mathrm{LN}(W_fF_t^{fine} + \mathrm{Up}(W_cF_t^{coarse}))
\]

Output:

```python
fused_grid: [B, T', Hf, Wf, 256]
```

Control baseline continues to use `dense_tokens`.

---

### 5.2 Thirty-two named local ACPR predicates

Copy and adapt the 32 predicate ontology from `FATE-OIA/acpr_calalign_v1_2`.

Required names:

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

Each predicate is not a Boolean bottleneck. It must contain:

```python
presence_logits: [B,T',32]
presence_probs: [B,T',32]
tokens: [B,T',32,256]
attention: [B,T',32,Hf,Wf]
trajectory_confidence: [B,T',32]
relative_motion: [B,T'-1,32,2]
```

This preserves the ACPR concept-embedding property: named and intervenable, but high-dimensional.

---

### 5.3 Local partial transport

Add a local partial transport module. Do not reuse the old global Sinkhorn as the primary method.

#### Coarse camera shift

For each adjacent time pair, score shifts over:

```text
dx, dy in [-2, -1, 0, 1, 2]
```

using mean cosine similarity on coarse features.

Use softmax over shift scores and compute expected shift:

```python
camera_shift: [B,T'-1,2]
```

#### Fine local matching

For every source patch, search a `5x5` neighborhood centered at:

```text
source_position + camera_shift
```

Add a dustbin candidate.

Use row-normalized partial transport:

\[
P(i,j)=
\mathrm{softmax}_{j\in \mathcal{N}(i)\cup \{\varnothing\}}
[
\cos(q_i,k_j)-\lambda\|p_j-\hat p_i\|^2
]
\]

Do not column-normalize.

Required output:

```python
local_transport_probs: [B,T'-1,N,26]  # 25 local positions + dustbin
dustbin_prob: [B,T'-1,N]
camera_shift: [B,T'-1,2]
```

For each predicate, transport the previous evidence distribution and compute predicate-specific confidence.

Formal complexity is local; no `N x N` dense matching tensor.

---

### 5.4 Temporal Predicate Embedding Field

For predicate `p` at first time step:

\[
A_{1,p} =
\mathrm{entmax}_{1.5}(q_p^\top X_1 + b_p^{region})
\]

At later time steps:

\[
A_{t,p} =
\mathrm{entmax}_{1.5}
[
q_p^\top X_t
+ \beta_{t,p}\log(\widetilde A_{t,p}+\epsilon)
+ b_p^{region}
]
\]

where:

- `transported prior` comes from local partial transport;
- `beta` depends on predicate-specific confidence;
- current visual evidence is always present;
- low confidence weakens the temporal prior.

Required dynamic descriptor for each predicate:

```python
now:         [B,32,256]
history:     [B,32,256]
trend:       [B,32,256]
volatility:  [B,32,256]
motion:      [B,32,2]
confidence:  [B,32,1]
descriptor:  [B,32,256]
```

Definitions:

\[
e_p^{now}=e_{T,p}
\]

\[
e_p^{history}=\mathrm{AttnPool}(e_{1:T,p})
\]

\[
e_p^{trend}=\sum_t \alpha_t(e_{t,p}-e_{t-1,p})
\]

\[
e_p^{volatility}=\mathrm{Mean}|e_{t,p}-2e_{t-1,p}+e_{t-2,p}|
\]

---

### 5.5 Factorized mesoscopic traffic-flow states

Do not use a flat mutually-exclusive 8-class traffic label.

Use thirteen named soft factors.

#### Regime

```text
clear_open_flow
stable_following
dense_following
queue_congestion
```

#### Phase

```text
forming
stable
releasing
oscillating
```

#### Source

```text
traffic_signal
lead_vehicle_group
merge_lane_constraint
turn_intersection
vulnerable_obstacle_conflict
```

Each flow factor query sparsely reads all 32 dynamic predicate descriptors.

\[
f_k = \mathrm{CrossAttn}_{entmax}(q_k,\{d_p\}_{p=1}^{32})
\]

Add a grammar prior from a YAML support matrix:

```python
factor_predicate_support: [13,32]
factor_predicate_contradiction: [13,32]
```

Outputs:

```python
flow_tokens: [B,13,256]
flow_logits: [B,13]
flow_probs: [B,13]
flow_to_predicate_attention: [B,13,32]
flow_evidence_maps: [B,T',13,Hf,Wf]
```

No hard traffic-flow annotations are required.

---

### 5.6 Online free-text supervision; no text cache

Do not build an image cache or a sentence-embedding cache.

Use raw action/justification strings from existing dataset metadata for deterministic PU text rules.

Use existing BERT word embeddings online to build a continuous reason target.

For text token embeddings:

```python
e_action = masked_mean(word_embeddings, action_token_mask)
e_reason = masked_mean(word_embeddings, reason_token_mask)
```

Action-residualize with per-sample Gram-Schmidt projection:

\[
r^* =
\mathrm{Norm}
[
e_R - \rho(e_R^\top e_A)e_A
]
\]

Use detached word embeddings for the target.

This removes much of the action-repetition direction without requiring an offline ridge artifact.

Required target outputs:

```python
reason_semantic_target: [B,768]
predicate_positive: [B,32]
predicate_contradiction: [B,32]
predicate_known_mask: [B,32]
predicate_reliability: [B,32]
flow_positive: [B,13]
flow_contradiction: [B,13]
flow_known_mask: [B,13]
flow_reliability: [B,13]
```

Unmentioned concepts are `unknown`, not hard negatives.

---

### 5.7 Free-text Positive–Unlabeled alignment

For named predicates and flow factors, use three states:

```text
positive
contradictory
unknown
```

Loss:

\[
L_{PU} =
-\sum_{Pos}\log p
-\sum_{Contra}\log(1-p)
-\eta\sum_{Unknown}\log(1-p)
\]

Default:

```yaml
unknown_negative_weight: 0.075
```

Explicit text evidence has reliability 1.0.

Control-derived weak phase evidence has reliability 0.5.

Unknown entries have low or zero weight.

The rule builder must be deterministic, unit-tested, and stored in YAML.

---

### 5.8 High-dimensional reason memory

Build both local and macro reason tokens.

#### Local reason tokens

For each named predicate:

\[
m_p^{local} =
\mathrm{LN}
(W_dd_p + W_ss_p)
\]

where `s_p` is a fixed semantic embedding derived from the predicate name through the local BERT word-embedding table.

#### Flow reason tokens

For each flow factor:

\[
m_k^{flow} =
\mathrm{LN}
(W_ff_k + W_p\sum_p a_{kp}d_p + W_ss_k)
\]

Add one learned null reason token.

Output:

```python
local_reason_memory: [B,32,768]
flow_reason_memory: [B,13,768]
null_reason_memory: [B,1,768]
reason_memory: [B,46,768]
reason_memory_mask: [B,46]
reason_memory_types: local / flow / null
global_reason_state: [B,768]
```

`ACPR-X` uses local predicate memory plus null and masks out flow memory.

`ACPR-FlowCal++` uses all memory tokens.

No flow state is allowed to bypass this reason memory.

---

### 5.9 Temporal HardPair

Implement a bounded queue-based adaptation of ACPR HardPair.

Queue fields:

```python
reason_target: [Q,768]
action_text_embedding: [Q,768]
global_video_embedding: [Q,256]
control_summary: [Q,8]
predicate_signature: [Q,32]
valid_control: [Q]
sample_id: list[str]
```

Default queue size:

```text
4096
```

A negative candidate is eligible when:

```text
action similarity >= 0.75
video similarity >= 0.45
control similarity >= 0.70 when control is valid
reason similarity <= 0.35
predicate contradiction or flow-source disagreement exists
```

Loss:

\[
L_{pair}
=
\max
[
0,
m-\cos(\hat r_i,r_i^*)
+\cos(\hat r_i,r_j^*)
]
\]

Defaults:

```yaml
margin: 0.20
max_pairs_per_batch: 64
start_epoch: 2
pair_weight: 0.03
pair_budget_ratio: 0.08
```

The weighted pair contribution must never exceed 8% of:

```text
action text loss + explanation text loss + control loss
```

Log:

```text
candidate_count
active_pair_count
active_pair_rate
no_pair_rate
margin_satisfied_rate
invalid_pair_rate
raw_pair_loss
budgeted_pair_loss
```

---

### 5.10 Temporal Sparse Evidence Co-Attention (Temporal SECA)

Delete the formal use of the current triple-product `TokenPMTAdapter`.

Apply SECA only to text hidden states, never to image hidden states.

For each text token:

\[
a_{j,k}
=
\mathrm{entmax}_{1.5}
[
(W_qh_j)^\top(W_km_k)/\sqrt d
]
\]

\[
c_j=\sum_ka_{j,k}W_vm_k
\]

\[
\Delta h_j=W_o c_j
\]

\[
h'_j=h_j+\gamma_{segment(j)}\tanh(\Delta h_j)
\]

Requirements:

- `W_o` Xavier initialized, not zero initialized;
- `gamma_action_raw = 0`;
- `gamma_explanation_raw = 0`;
- bounded gate:
  ```python
  gamma = max_scale * tanh(gamma_raw)
  ```
- max action scale `0.15`;
- max explanation scale `0.30`;
- separate gates;
- a null reason token;
- entmax attention;
- output routing:
  ```python
  token_reason_attention: [B,L_text,46]
  token_delta: [B,L_text,768]
  ```

#### Controlled gradient bridge

For action and control reading:

```python
memory_forward_equal = memory
memory_backward_gradient = evidence_grad_scale * full_gradient
```

Implement:

```python
memory_scaled = memory.detach() + scale * (memory - memory.detach())
```

Defaults:

```text
action evidence gradient scale = 0.25
control evidence gradient scale = 0.25
explanation evidence gradient scale = 1.00
```

This preserves the forward value while preventing action/control gradients from overwhelming reason learning.

#### BERT integration point

Patch the actual `BertForImageCaptioning` implementation.

Required order:

```text
BERT multimodal hidden states
-> split text hidden and visual hidden
-> Temporal SECA on text hidden only
-> recombine
-> MLM prediction head
```

Do not modify image hidden states through SECA.

Training text length:

```python
text_len = masked_pos.shape[-1]
```

Decode text length:

```python
text_len = input_ids.shape[-1]
```

Beam generation must expand reason memory to beam batch size.

---

### 5.11 Reason-mediated control adapter

Modify the sensor head backward-compatibly to expose:

```python
base_control_prediction: [B,32,2]
control_hidden: [B,32,768]
```

For each control timestep, query the same reason memory using entmax.

\[
c_t^U = \mathrm{Attn}_{entmax}(h_t^U,M_R)
\]

\[
\Delta U_t = \tanh(W_Uc_t^U)\odot(0.15\sigma_U)
\]

\[
U_t=U_t^{ADAPT}+\gamma_U\Delta U_t
\]

Requirements:

- one bounded ReZero gate per signal;
- gate zero exactly recovers ADAPT control;
- only reason memory is read;
- no direct flow-to-control residual;
- invalid `-1` control labels are masked in loss and metrics;
- base and enhanced control predictions are both logged.

---

### 5.12 Prefix-to-future auxiliary task

On 25% of training batches only:

```text
prefix frames: first 24 of 32
target control: final 8 steps
```

The prefix state branch predicts:

```python
future_speed_delta
future_course_delta
```

Future signals are targets only, never state inputs.

Loss weight:

```yaml
future_control: 0.02
```

This ensures traffic-flow state carries predictive decision information rather than only describing an already-observed action.

This auxiliary is disabled in `ACPR-X` and enabled in the full model.

---

### 5.13 Sequence-CalAlign

Sequence-CalAlign is the BDD-X analogue of ACPR-CalAlign.

It does not replace model training. It selects deployment strengths for learned residuals.

Store:

```python
alpha_action
alpha_explanation
alpha_speed
alpha_course
temperature_action
temperature_explanation
```

All candidate grids must include zero, so ADAPT is always in the deployment feasible set.

Default grids:

```yaml
alpha_action: [0.0, 0.025, 0.05, 0.075, 0.10, 0.15]
alpha_explanation: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
alpha_speed: [0.0, 0.05, 0.10, 0.15, 0.20]
alpha_course: [0.0, 0.05, 0.10, 0.15, 0.20]
temperature_action: [0.85, 0.925, 1.0, 1.075, 1.15]
temperature_explanation: [0.85, 0.925, 1.0, 1.075, 1.15]
```

Fit only on a deterministic 10% `train_calib` subset.

Main model parameters are frozen and detached during calibration.

No test metric may update:

```text
alpha
temperature
model parameters
learning rate
checkpoint weights
```

Formal test still selects the best epoch as requested, but test labels never fit Sequence-CalAlign.

---

## 6. Named output contracts

Add typed dataclasses.

### `ACPRFlowBundle`

At minimum:

```python
fine_grid
coarse_grid
fused_grid
camera_shift
local_transport_probs
transport_dustbin
predicate_attention
predicate_tokens_temporal
predicate_logits_temporal
predicate_probs_temporal
predicate_confidence
predicate_relative_motion
predicate_descriptor
flow_tokens
flow_logits
flow_probs
flow_to_predicate_attention
flow_evidence_maps
local_reason_memory
flow_reason_memory
reason_memory
reason_memory_mask
global_reason_state
token_reason_attention
token_delta
control_reason_attention
control_delta
diagnostics
```

### `ACPRFlowTrainOutput`

At minimum:

```python
action_text_loss
explanation_text_loss
text_loss_total
control_loss
control_base_prediction
control_final_prediction
baseline_masked_logits
enhanced_masked_logits
auxiliary_loss
total_loss
loss_components
bundle
```

Formal trainers must not parse positional tuple tails.

The legacy ADAPT API may remain available when `return_acpr_flow_output=false`.

---

## 7. Files to add

```text
configs/acpr_flow_local_predicates.yaml
configs/acpr_flow_factors.yaml
configs/acpr_flow_text_rules.yaml
configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml

fate_x/acpr_flow/__init__.py
fate_x/acpr_flow/types.py
fate_x/acpr_flow/region_priors.py
fate_x/acpr_flow/local_partial_transport.py
fate_x/acpr_flow/temporal_predicate_field.py
fate_x/acpr_flow/flow_factor_composer.py
fate_x/acpr_flow/free_text_partial_targets.py
fate_x/acpr_flow/online_reason_target.py
fate_x/acpr_flow/reason_memory.py
fate_x/acpr_flow/temporal_seca.py
fate_x/acpr_flow/reason_control_adapter.py
fate_x/acpr_flow/prefix_future_head.py
fate_x/acpr_flow/temporal_hard_pair.py
fate_x/acpr_flow/sequence_calalign.py
fate_x/acpr_flow/interventions.py
fate_x/acpr_flow/model.py

fate_x/losses/acpr_flowcal_losses.py
fate_x/losses/segment_caption_loss.py

fate_x/engine/train_acpr_flowcal_pp.py
fate_x/engine/eval_acpr_flowcal_pp.py
fate_x/engine/fit_sequence_calalign.py
fate_x/engine/audit_acpr_flowcal_pp.py
fate_x/engine/probe_acpr_flowcal_memory.py
fate_x/engine/supervise_acpr_flowcal_foreground.py
fate_x/engine/export_acpr_flow_visuals.py
fate_x/engine/build_acpr_flow_atlas.py

fate_x/explain/acpr_flow_renderer.py
fate_x/explain/acpr_flow_atlas.py
fate_x/explain/acpr_flow_faithfulness.py

fate_x/utils/acpr_flow_artifacts.py
fate_x/utils/acpr_flow_config.py
fate_x/utils/acpr_flow_git_guard.py

scripts/FATE_X_acpr_flowcal_pp_v1_foreground.ps1
scripts/FATE_X_acpr_flowcal_pp_v1_foreground.sh

.codex/skills/acpr-flowcal-pp-implementation-audit/SKILL.md
docs/runbooks/ACPR_FlowCalPP_V1_Implementation_Plan.md
```

---

## 8. Files to modify

Modify only as required:

```text
src/modeling/load_swin.py
src/modeling/video_swin/swin_transformer.py
src/modeling/load_sensor_pred_head.py
src/layers/bert/modeling_bert.py
src/datasets/vision_language_tsv.py
src/datasets/vl_dataloader.py
fate_x/models/__init__.py
.gitignore
```

The formal trainer should be new and typed. Do not continue expanding `src/tasks/run_adapt.py` for the formal ACPR-FlowCal++ experiment.

`run_adapt.py` remains the ADAPT baseline entrypoint.

---

## 9. Dataset metadata and collate contract

Existing dataset metadata already contains raw caption data and `img_key`.

Harden it to return explicit fields:

```python
meta_data = {
    "sample_id": img_key,
    "raw_action": action_string,
    "raw_justification": justification_string,
    "is_video": is_video,
}
```

Add a custom collate function that preserves strings as lists and collates numeric tensors normally.

Formal training batch:

```python
ACPRFlowBatch(
    input_ids,
    attention_mask,
    token_type_ids,
    frames,
    masked_pos,
    masked_ids,
    car_info,
    sample_ids,
    raw_actions,
    raw_justifications,
)
```

No fragile interpretation of default-collated tuple strings.

---

## 10. Losses

Use:

\[
\begin{aligned}
L={}&
L_A
+L_E
+0.05L_U\\
&+0.05L_{predicate\_PU}
+0.03L_{flow\_PU}
+0.05L_{reason\_semantic}\\
&+0.02L_{future}
+L_{pair}^{budgeted}\\
&+0.01L_{trajectory}
+0.02L_{action\_preserve}
+0.01L_{control\_preserve}\\
&+0.001L_{memory\_diversity}
\end{aligned}
\]

Notes:

- `L_A`: action-segment token loss;
- `L_E`: explanation-segment token loss;
- `L_U`: masked speed/course loss;
- no latent-vector intervention loss;
- intervention is evaluation-only;
- no legacy sparse-mask loss;
- pair loss is budgeted, not double-added.

All loss components must be separately logged.

---

## 11. Optimizer and schedules

Use native PyTorch AdamW and BF16 on the single 48GB GPU.

Do not use DeepSpeed in the formal path unless BF16 native execution fails for an independently verified environment reason.

Runtime BF16 check:

```python
torch.cuda.is_bf16_supported()
```

Fallback to FP16 only if unsupported, with:

```text
GradScaler initial scale <= 4096
finite-loss/finite-gradient guard
overflow counter
```

### Parameter groups

```yaml
predicate_field_lr: 1.0e-4
flow_composer_lr: 1.0e-4
reason_memory_lr: 1.0e-4
hard_pair_projection_lr: 5.0e-5
temporal_seca_lr: 5.0e-5
control_adapter_lr: 2.0e-5
future_head_lr: 5.0e-5
bert_last2_lr: 1.0e-5
swin_last_stage_lr: 5.0e-6
sensor_head_lr: 1.0e-5
```

Weight decay:

```yaml
new_modules: 0.01
backbone: 0.05
bias_and_norm: 0.0
```

Scheduler:

```yaml
scheduler: cosine
warmup_ratio: 0.05
min_lr_ratio: 0.05
gradient_clip_norm: 1.0
```

Optimizer manifest must list every parameter exactly once.

---

## 12. Formal experiment suite

The experiment suite is progressive and fair.

### Run 0 — ADAPT baseline evaluation

No training.

Evaluate the released ADAPT checkpoint on the same test split using:

```text
action narration metrics
justification metrics
speed RMSE
course RMSE
```

Save frozen baseline metrics and predictions.

### Common Stage A — ACPR-X initialization

Train six epochs.

Model:

```text
32 named local predicates
online PU free-text supervision
local predicate reason memory
Temporal HardPair
Temporal SECA
reason-mediated control adapter
```

Disabled:

```text
local temporal transport
factorized flow tokens
prefix-to-future loss
```

Schedule:

```text
epochs 0-1:
  freeze Video Swin and BERT
  action/control SECA gate max = 0
  explanation gate ramps 0 -> 0.05

epochs 2-5:
  unfreeze BERT last 2 layers
  unfreeze Video Swin final stage
  action gate max = 0.08
  explanation gate max = 0.20
  control gate max = 0.08
  HardPair enabled
```

Save:

```text
checkpoint_acpr_x_common/
```

### Fork B1 — ACPR-X equal-budget continuation

Resume the exact common checkpoint.

Train 12 additional epochs with transport/flow/future still disabled.

Then run 3 calibration epochs.

Total ACPR-X exposure:

```text
6 + 12 + 3 = 21 epochs
```

### Fork B2 — ACPR-FlowCal++ full continuation

Resume the exact same common checkpoint.

Enable:

```text
local partial transport
factorized flow states
flow reason memory
prefix-to-future auxiliary
flow-aware HardPair signatures
```

Train 12 additional epochs.

Then run 3 calibration epochs.

Total full exposure:

```text
6 + 12 + 3 = 21 epochs
```

B1 and B2 therefore receive equal training budget after the same common checkpoint.

### No-retrain full-checkpoint analyses

Run on B2 best checkpoint:

```text
Sequence-CalAlign alpha = 0
local transport off
flow tokens off
temporal shuffle
temporal reverse
top flow state off
top local predicate off
top evidence tube off
random equal-mass off
prefix lengths 8 / 16 / 24 / 32
```

These are real re-forward interventions.

### Optional journal ablation

Only if requested after the full result:

```text
B2 w/o HardPair
```

Do not automatically add another long run before seeing the two primary results.

---

## 13. Per-epoch evaluation and best selection

The user-requested protocol is test-only.

No validation loader in formal runs.

At the end of every epoch:

1. fit/update Sequence-CalAlign only from train-calib;
2. evaluate test exactly once;
3. evaluate action narration;
4. evaluate original justification;
5. evaluate speed/course;
6. run bounded fixed-case intervention diagnostics;
7. save latest;
8. update test-best.

Primary eligibility:

```text
speed RMSE <= ADAPT speed RMSE * 1.01
course RMSE <= ADAPT course RMSE * 1.01
```

Among eligible checkpoints, maximize:

```text
CIDEr_action + CIDEr_explanation
```

Tie breakers:

```text
higher CIDEr_explanation
higher METEOR_explanation
higher CIDEr_action
lower speed RMSE
lower course RMSE
```

If no enhanced configuration is control-eligible, Sequence-CalAlign must be allowed to set control alpha to zero.

Save:

```text
checkpoint_latest/
checkpoint_best_test/
checkpoint_best_action/
checkpoint_best_explanation/
checkpoint_best_control_safe/
```

All manifests must state:

```text
protocol_tag = test_selected_user_requested
```

No metric-based early stopping.

---

## 14. Memory plan for RTX 5880 48GB

Formal target:

```text
preferred peak reserved: 30-42 GiB
hard limit: 43 GiB
```

Do not allocate useless memory to reach 30 GiB.

Probe:

```text
micro-batch 5 / accumulation 13 -> effective 65
micro-batch 4 / accumulation 16 -> effective 64
micro-batch 3 / accumulation 22 -> effective 66
micro-batch 2 / accumulation 32 -> effective 64
```

For each candidate:

```text
3 warmup steps
20 measured full forward/backward steps
all formal losses enabled for the current stage
direct image tensors
no cache
BF16
gradient checkpointing
```

Select the largest stable micro-batch satisfying:

```text
max_memory_reserved <= 43 GiB
no OOM
no NaN/Inf
no skipped optimizer step
```

Record:

```text
peak allocated
peak reserved
step time
images/sec
selected micro-batch
selected accumulation
effective batch
```

---

## 15. Intervention implementation

Use immutable `InterventionSpec`.

Supported:

```python
None
StateOff(flow_factor_idx)
PredicateOff(predicate_idx)
EvidenceTubeOff(memory_idx, equal_mass=False)
RandomEqualMass(memory_idx, seed)
TemporalShuffle(seed)
TemporalReverse()
PrefixLength(n_frames)
```

### State off

Zero the selected flow token and recompute:

```text
reason memory
global reason state
SECA routing
action/explanation
control residual
```

### Predicate off

Remove the selected predicate descriptor and recompute:

```text
flow
reason memory
outputs
```

### Evidence tube off

Use the selected predicate/flow evidence map to replace the corresponding fused-grid feature cells with the framewise spatial mean.

Then recompute:

```text
predicate field
flow
reason memory
outputs
```

Video Swin need not rerun.

### Random equal mass

Use the same per-frame patch count or cumulative evidence mass.

### Temporal shuffle/reverse

Reorder fused-grid time and recompute the complete state branch.

Do not mutate a previously used bundle in place.

---

## 16. Faithfulness metrics

Teacher-forced metrics:

\[
SIS_A(k)=NLL_A(do(k=0))-NLL_A
\]

\[
SIS_E(k)=NLL_E(do(k=0))-NLL_E
\]

Evidence Tube Fidelity:

\[
ETF=
\Delta NLL_{evidence}
-
E[\Delta NLL_{random\ equal\ mass}]
\]

Action–Explanation Mediation Agreement:

\[
AEMA=
Spearman([SIS_A(k)]_k,[SIS_E(k)]_k)
\]

Control State Intervention:

\[
CSI(k)=
\|U-U_{do(k=0)}\|
\]

Temporal Direction Sensitivity:

\[
TDS=
NLL(Reverse(V))-NLL(V)
\]

These are model-level intervention metrics, not claims of real-world causal effect.

---

## 17. Visualization

### 17.1 ACPR-Flow Canvas

Each selected sample must automatically contain:

1. **Named predicate tubes**
   - same predicate uses the same color across frames;
   - overlays on de-normalized input frames.

2. **Factorized flow panel**
   - top regime;
   - top phase;
   - top source;
   - activation, trend, volatility, confidence.

3. **Hierarchical support graph**
   ```text
   video region -> local predicate -> flow factor -> reason memory
   -> action/explanation/control
   ```

4. **Token effect panel**
   - real teacher-forced state-off log-prob changes for action and explanation tokens.

5. **Control panel**
   - ADAPT base;
   - enhanced prediction;
   - state-off prediction;
   - GT where valid.

6. **Counterfactual twin**
   - factual free generation;
   - top-state-off free generation;
   - no templated or fabricated sentence.

### 17.2 Dataset Traffic-Flow Decision Atlas

Aggregate full test:

```text
regime -> action effect matrix
phase -> action effect matrix
source -> explanation effect matrix
flow -> speed/course effect matrix
flow transition graph
prototype justification examples
representative predicate tubes
confidence/effect distributions
failure cases
```

No manual boxes.

---

## 18. Artifact schema

Every epoch:

```text
epoch_XXX/
  caption_metrics_action.json
  caption_metrics_explanation.json
  control_metrics.json
  deployment_metrics.json
  loss_components.jsonl
  optimizer_groups.json
  gradient_norms.json
  predicate_stats.json
  flow_stats.json
  reason_stats.json
  hardpair_stats.json
  seca_stats.json
  sequence_calalign.json
  intervention_stats.json
  faithfulness_summary.json
  predictions.jsonl
  token_logprobs.jsonl
  state_effects.jsonl
  failure_cases.jsonl
  visuals_index.jsonl
  visuals/
```

Run root:

```text
config_resolved.yaml
config_resolved.json
run_manifest.json
implementation_manifest.json
review_report.json
memory_probe.json
supervisor_live_status.json
supervisor_decisions.jsonl
train.log
checkpoint_latest/
checkpoint_best_test/
run_complete.json
```

Missing values must be marked unavailable with an explicit reason. Do not write fake zeros.

---

## 19. Foreground supervisor contract

Add:

```text
fate_x/engine/supervise_acpr_flowcal_foreground.py
scripts/FATE_X_acpr_flowcal_pp_v1_foreground.ps1
```

PowerShell must synchronously enter WSL and remain attached.

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

Supervisor sequence:

```text
verify review pass SHA
verify branch
verify clean worktree
verify remote SHA
baseline evaluation
memory probe
Common Stage A
Fork B1
B1 Sequence-CalAlign
Fork B2
B2 Sequence-CalAlign
full no-retrain interventions
Canvas generation
Atlas generation
final summaries
```

Behavior:

- heartbeat every 60 seconds;
- stream child stdout/stderr;
- verify checkpoint and metrics after every epoch;
- never stop because metrics are low or flat;
- OOM: choose next fallback batch and resume;
- transient I/O/eval error: retry up to three times;
- reproducible code error: remain in foreground repair workflow;
- after a code patch:
  ```text
  test -> commit -> push -> full audit -> new review pass -> resume
  ```
- stop only for:
  1. explicit user stop sentinel;
  2. unrecoverable hardware loss requiring the user;
  3. completion of the formal suite.

User sentinel:

```text
.background_runs/<suite>/control/STOP_REQUESTED_BY_USER
```

Codex must not create it without an explicit user command.

---

## 20. Tests

Add at least:

```text
tests/test_acpr_flow_named_predicates.py
tests/test_acpr_flow_region_priors.py
tests/test_acpr_flow_local_partial_transport.py
tests/test_acpr_flow_predicate_trajectories.py
tests/test_acpr_flow_dynamic_descriptors.py
tests/test_acpr_flow_factor_composer.py
tests/test_acpr_flow_free_text_targets.py
tests/test_acpr_flow_online_reason_target.py
tests/test_acpr_flow_reason_memory.py
tests/test_acpr_flow_temporal_seca.py
tests/test_acpr_flow_control_adapter.py
tests/test_acpr_flow_hardpair.py
tests/test_acpr_flow_pair_budget.py
tests/test_acpr_flow_sequence_calalign.py
tests/test_acpr_flow_interventions.py
tests/test_acpr_flow_live_decode.py
tests/test_acpr_flow_visuals.py
tests/test_acpr_flow_named_output.py
tests/test_acpr_flow_config_contract.py
tests/test_acpr_flow_optimizer_groups.py
tests/test_acpr_flow_test_only_protocol.py
tests/test_acpr_flow_supervisor_foreground.py
tests/test_acpr_flow_e2e_direct_image_smoke.py
```

Mandatory assertions include:

```text
32 exact predicate names
13 exact flow factor names
local transport row mass plus dustbin = 1
global translation compensation
predicate temporal continuity
low-confidence prior weakening
trend sign changes under temporal reverse
flow evidence map equals weighted predicate map composition
unknown free-text labels are not hard negatives
online action-residual reason target finite
reason memory includes local, flow, null
flow cannot bypass reason memory
SECA touches text hidden only
zero gate equals ADAPT
zero gate has nonzero gate gradient
after one step q/k/v/out gradients are nonzero
action gradient into reason memory is scaled by 0.25
control zero gate equals ADAPT
HardPair active path and budget
Sequence-CalAlign zero alpha equals ADAPT
test cannot update calibration
real interventions alter downstream output
equal-mass random is truly equal mass
beam size 3 expands reason memory correctly
no feature cache
no validation loader
no positional tuple parsing
no legacy PMT in formal import graph
foreground supervisor is attached
```

---

## 21. Preflight stages

### Gate A — static and unit

```bash
python -m compileall -q fate_x src
python -m pytest tests/test_acpr_flow_*.py -q
python -m pytest tests -q
```

### Gate B — real direct-image 8-step smoke

```text
train samples: >= 8
test samples: >= 8
batch: 1
steps: 8
beam: 1
```

Require:

```text
direct image tensor [B,32,3,224,224]
no cache
finite loss
finite logits
checkpoint latest
test evaluation
```

### Gate C — gradient chain

On a real batch require finite nonzero gradients for:

```text
predicate queries
local transport feature projection
flow queries
reason memory projection
global reason state
SECA action gate
SECA explanation gate
control gate
HardPair projection when pair active
```

At gate zero:

```text
SECA gate grad > 0
```

After one optimizer step:

```text
SECA q/k/v/out grad > 0
```

### Gate D — 128-sample mechanism overfit

Train on 128 samples long enough to demonstrate:

```text
action token loss decreases
explanation token loss decreases
control loss decreases on valid rows
reason semantic cosine improves
predicate PU loss decreases
flow weak loss decreases in full mode
reason memory does not collapse
HardPair active pair rate > 0
```

This is a mechanism test, not a performance claim.

### Gate E — temporal necessity

Using one checkpoint:

```text
normal
reverse
shuffle
last frame only
```

Require:

- static predicate presence may remain partly stable;
- flow phase/trend changes materially;
- prefix-to-future prediction worsens under shuffle/reverse.

### Gate F — real intervention

Require:

```text
state-off action or explanation NLL delta != 0
evidence deletion effect > equal-mass random on average
temporal reverse changes phase and output
control state intervention is finite
```

### Gate G — implementation audit

Only then run:

```bash
python -m fate_x.engine.audit_acpr_flowcal_pp \
  --config configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml \
  --output_dir .background_runs/acpr_flowcal_pp_v1_preflight \
  --device cuda \
  --write_review_pass
```

Required:

```text
.background_runs/acpr_flowcal_pp_v1_preflight/
REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt
```

The pass file must bind the exact clean pushed SHA.

---

## 22. Milestones

```text
M0  Read context, inspect worktree, snapshot existing dirty code
M1  Agent B approves implementation manifest
M2  Dedicated typed trainer and named outputs
M3  Hardened metadata/collate and online free-text targets
M4  Local partial transport
M5  32 named temporal predicate embeddings
M6  Factorized flow composer
M7  High-dimensional reason memory
M8  Temporal SECA and real generation integration
M9  Reason-mediated control adapter
M10 Temporal HardPair and budget
M11 Prefix-to-future auxiliary
M12 Sequence-CalAlign
M13 Real interventions
M14 Canvas and Atlas
M15 Unit/regression/smoke/overfit/temporal/intervention gates
M16 Commit, push, audit exact HEAD
M17 Foreground formal experiment suite
M18 Final artifact and GitHub verification
```

---

## 23. GitHub synchronization

Before formal training:

```powershell
git status --porcelain
git add <code/config/tests/scripts/skill/runbook only>
git commit -m "Implement ACPR-FlowCal++ dynamic predicate reasoning for BDD-X"
git push github flowtrace_pmt_v1:flowtrace_pmt_v1
git rev-parse HEAD
git ls-remote github refs/heads/flowtrace_pmt_v1
```

Local and remote SHA must match.

Do not commit run outputs.

---

## 24. Definition of implementation complete

Codex may state implementation complete only when:

```text
formal path uses named outputs
formal path does not import old PMT/Sinkhorn/placeholder renderer
32 named predicate trajectories execute
13 factorized flow states execute
online free-text PU targets execute
reason semantic target executes
Temporal HardPair executes and is budgeted
Temporal SECA executes pre-LM-head on text only
control residual reads reason memory only
zero-scale output equals ADAPT
gradient chain is non-dead
Sequence-CalAlign is train-calib only
real interventions re-forward downstream components
Canvas and Atlas contain real values
direct-image/no-cache proof exists
test-only protocol exists
memory probe exists
foreground supervisor is attached
all tests pass
real smoke passes
128-sample mechanism test passes
review pass exists for exact SHA
branch is pushed and clean
```

File existence alone is insufficient.

---

## 25. Definition of formal experiment complete

The suite is complete only after:

```text
ADAPT baseline test evaluation
6-epoch common ACPR-X stage
12+3 epoch ACPR-X continuation/calibration
12+3 epoch ACPR-FlowCal++ continuation/calibration
test evaluation after every epoch
best-test checkpoints
speed/course metrics
action narration metrics
original justification metrics
faithfulness metrics
fixed-case Canvas
full-test Atlas
run_complete.json
canonical task_plan/findings/progress updates
```

---

## 26. Hard audit failures

Training must remain blocked if any holds:

```text
reason supervision disconnected
flow can modify action/control without reason memory
SECA modifies image hidden states
SECA output projection is zero initialized
zero-gate gate gradient is zero
positional tuple parsing remains in formal trainer
global Sinkhorn is used as formal transport
mean-only state memory
GT text enters inference
HardPair is configured but silent
pair loss exceeds budget
unknown free-text concepts are treated as full negatives
control uses state tokens directly
intervention only changes a display tensor
renderer is heatmap-only
Atlas is a record dump
feature cache is built/read
learn_mask is enabled
validation loader is created
test updates calibration/model
formal config fields are unused
BF16/FP16 nonfinite source is ignored
supervisor detaches or metric-stops
worktree is dirty
local and remote SHA differ
review pass SHA differs
```

---

## 27. Formal launch command

After review pass:

```powershell
cd E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree

powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts\FATE_X_acpr_flowcal_pp_v1_foreground.ps1 `
  -Config configs\acpr_flowcal_pp_v1_bddx_32f_224.yaml `
  -RequireReviewPass
```

The launcher must remain attached.

---

## 28. Required reporting boundaries

Can claim only after results support it:

```text
static ACPR predicate reasoning transfers from BDD-OIA to BDD-X
video trajectories create decision-relevant traffic-flow states
reason memory mediates action, explanation, and control
Sequence-CalAlign protects the ADAPT fallback
model-level intervention traces the decision chain
```

Do not claim:

```text
real-world causal effect
global road-network traffic density
traffic-flow ground-truth accuracy
paper-level superiority from smoke
faithfulness from attention alone
unbiased test performance when test selects best
```
