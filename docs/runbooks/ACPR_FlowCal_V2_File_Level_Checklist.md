# ACPR-FlowCal++ V2 file-level implementation checklist

This checklist is subordinate to the full V2 plan. It exists to make omissions mechanically visible.

## A. Existing files that must be modified

| Existing file | Required change | Prohibited shortcut | Required tests |
|---|---|---|---|
| `src/modeling/load_sensor_pred_head.py` | Add target-independent `encode()` and `predict(frame_num)`; retain backward-compatible `forward`; expose hidden; circular loss remains outside predictor | Using `car_info` values as encoder input; replacing with a new uninitialized local linear head | `test_v2_adapt_motion_equivalence.py`, `test_v2_motion_target_independence.py` |
| `src/layers/bert/modeling_bert.py` | Add V2 typed hook before LM head; propagate V2 bundle/readers through `prepare_inputs_for_generation`; implement explanation generation with token log-probabilities for SCST | Reusing V1 hook arguments as the V2 API; modifying image hidden; fabricated/replay log-probs | `test_v2_seca_segment_readers.py`, `test_v2_scst_logprob_reward.py`, `test_v2_zero_gate_fallback.py` |
| `src/modeling/load_swin.py` | Guarantee stage-return API used by V2 and record load report | A second Video-Swin forward for reasoning | `test_v2_adapt_video_load.py` |
| `src/modeling/video_swin/swin_transformer.py` | Preserve ordinary ADAPT output and support stable `return_stages=True` contract if current implementation is insufficient | Changing baseline default behavior | ADAPT regression + V2 backbone test |
| `src/datasets/vision_language_tsv.py` | Preserve raw action, justification, and sample ID in metadata for V2; no cache | Re-tokenizing away original text; reading feature cache | `test_v2_adapt_text_contract.py` |
| `src/datasets/vl_dataloader.py` | Return V2-preserved metadata and deterministic test order; train-calib ID hashing support through V2 wrapper | Creating a val loader for formal V2 | `test_v2_test_only_protocol.py` |
| `src/tasks/run_adapt.py` | Only add narrowly scoped baseline-evaluation compatibility if required; do not make it the V2 formal trainer | Reintroducing positional tuple parsing or V1 PMT | import-graph test |
| `.gitignore` | Ensure active runs, checkpoints, logits, caches, visuals, review artifacts remain untracked | Ignoring source package directories | git audit |

## B. New package files and exact public symbols

### `fate_x/acpr_flow_v2/config.py`

Required:

```python
@dataclass(frozen=True)
class FlowCalV2Config: ...
def load_flowcal_v2_config(path: str | Path) -> FlowCalV2Config
def write_resolved_config(config, output_path) -> None
def build_config_binding_manifest(config) -> dict
```

Requirements:

- nested dataclasses;
- reject unknown keys unless explicit allowlist;
- reject total reason tokens other than 54;
- reject formal cache flags;
- validate nonoverlapping stage epochs 0–14;
- validate every stage train/freeze name.

### `fate_x/acpr_flow_v2/types.py`

Required dataclasses:

```text
FlowCalV2Batch
VideoBackboneOutput
LocalTransportOutput
PredicateTrajectory
LaneFlowFieldOutput
AxisAwareFlowOutput
SemanticReasonMemory
FlowCalV2Bundle
FlowCalV2TrainOutput
GeneratedSequence
InterventionSpecV2
```

All fields named and documented with shapes.

### `adapt_video_backbone.py`

Required:

```python
class ADAPTVideoBackboneV2(nn.Module):
    def forward(self, frames) -> VideoBackboneOutput
    def reset_forward_counter(self) -> None
```

Private helpers:

```text
_extract_native_stages
_temporal_align
_project_dense_tokens
_fuse_reasoning_grids
```

Must load Video-Swin and ADAPT fc separately and report keys.

### `adapt_motion_backbone.py`

Required:

```python
class ADAPTMotionBackbone(nn.Module):
    @classmethod
    def from_adapt_checkpoint(...)
    def predict(self, dense_tokens, steps) -> tuple[Tensor, Tensor]
```

Do not calculate training loss inside `predict`.

### `local_partial_transport.py`

Required:

```python
class LocalPartialTransportV2(nn.Module):
    def forward(self, fused_grid, coarse_grid=None) -> LocalTransportOutput

def warp_source_map_to_current(...)
def expected_transport_displacement(...)
```

### `temporal_predicate_tracker.py`

Required:

```python
class TransportedNamedPredicateTracker(nn.Module):
    def forward(
        self,
        fused_grid: Tensor,
        transport: LocalTransportOutput,
    ) -> PredicateTrajectory
```

Expose diagnostic beta, transported mass, entropy, current-score contribution.

### `lane_flow_field.py`

Required:

```python
class PredicateConditionedLaneFlowField(nn.Module):
    def forward(
        self,
        predicates: PredicateTrajectory,
        fused_grid: Tensor,
    ) -> LaneFlowFieldOutput
```

Required helper:

```text
build_soft_corridor_masks
refine_masks_with_drivable_predicates
aggregate_region_statistics
temporal_encode_regions
```

### `axis_aware_flow_composer.py`

Required:

```python
class AxisAwareFlowComposer(nn.Module):
    def forward(
        self,
        predicates: PredicateTrajectory,
        lane_flow: LaneFlowFieldOutput,
    ) -> AxisAwareFlowOutput

def derive_axis_direction_targets(
    control_targets,
    signal_names,
    control_stats,
) -> dict[str, Tensor]
```

Control targets never enter `forward`.

### `contextual_reason_target.py`

Required:

```python
class FrozenContextualReasonTarget(nn.Module):
    def encode_texts(self, texts: list[str]) -> Tensor
    def build_target(actions, justifications) -> dict[str, Tensor]

class ActionSubspaceTracker:
    def update(self, action_embeddings) -> None
    def finalize_epoch(self) -> None
    def state_dict(self) -> dict
    def load_state_dict(self, state) -> None
```

Target module must be absent from inference call graph.

### `pu_targets.py`

Required:

```python
class FreeTextPUTargetBuilderV2:
    @classmethod
    def from_yaml(...)
    def build(actions, justifications, epoch) -> PUTargetBatch

def positive_unlabeled_loss_v2(...)
```

### `semantic_reason_memory.py`

Required:

```python
class SemanticReasonMemoryBuilder(nn.Module):
    def forward(
        self,
        predicates,
        lane_flow,
        flow_state,
    ) -> SemanticReasonMemory

def longitudinal_memory_mask(memory) -> Tensor
def lateral_memory_mask(memory) -> Tensor
```

### `semantic_gradient_firewall.py`

Required:

```python
def scaled_gradient(x, scale): ...
def representation_pcgrad_surrogate(
    reason_memory,
    semantic_loss,
    control_loss,
) -> tuple[Tensor, dict[str, Tensor]]
```

The returned surrogate must have approximately zero forward scalar value after centering, but the intended correction gradient.

### `temporal_seca.py`

Required:

```python
class TemporalSECAV2(nn.Module):
    def forward(
        self,
        hidden,
        memory: SemanticReasonMemory,
        token_type_ids,
        text_len,
        generation_segment=None,
    ) -> tuple[Tensor, SECADiagnostics]
```

Separate action/explanation query/readout parameters.

### `axis_aware_control_adapter.py`

Required:

```python
class AxisAwareReasonControlAdapter(nn.Module):
    def forward(
        self,
        base_prediction,
        control_hidden,
        memory,
        control_stats,
    ) -> AxisControlOutput
```

Map signal indices by names, never assume speed is index 0.

### `temporal_hardpair.py`

Required:

```python
class ContradictionAwareTemporalHardPair(nn.Module):
    def mine(...)
    def forward(...)
    def enqueue(...)
```

Queue state must not update during eval.

### `prefix_future.py`

Required:

```python
class PrefixFuturePredictor(nn.Module): ...
def build_prefix_bundle_from_precomputed_grids(...)
```

Must not call Video-Swin again.

### `sequence_calalign.py`

Required:

```python
@dataclass
class SequenceCalAlignV2Scales: ...
class SequenceCalAlignV2:
    def fit_text(...)
    def fit_control(...)
    def apply_text(...)
    def apply_control(...)
```

Fit all branches; course uses circular residual.

### `interventions.py`

Required:

```python
class FlowCalV2InterventionEngine:
    def rerun_from_visual(...)
    def rerun_from_predicates(...)
    def rerun_from_flow(...)
    def rerun_from_memory(...)
```

Each `InterventionSpecV2.kind` maps to the earliest affected layer.

### `model.py`

Required:

```python
class ACPRFlowCalV2Model(nn.Module):
    def build_visual_state(...)
    def build_reason_state(...)
    def forward_text(...)
    def forward_control(...)
    def forward(...)
    def decode_adapt_compatible(...)
    def generate_explanation_with_logprobs(...)
```

`forward()` must orchestrate but not bury all implementation in one 1000-line file.

## C. New losses

### `fate_x/losses/acpr_flowcal_v2_losses.py`

Required:

```text
shortest_circular_delta
normalized_control_huber
transport_consistency_loss
lane_temporal_consistency_loss
axis_direction_weak_loss
delta_kl_loss
parameter_anchor_loss
memory_diversity_loss
```

### `fate_x/losses/explanation_scst.py`

Required:

```text
sentence_cider_reward
sentence_meteor_reward
hallucination_penalty
self_critical_explanation_loss
```

## D. Data/engine files

### `acpr_flowcal_v2_data.py`

Required:

```python
def resolve_adapt_text_contract(...)
def build_v2_dataloader(split: Literal["train","test"], ...)
def adapt_batch_to_v2(...)
def stream_train_control_stats(...)
def deterministic_train_calib_ids(...)
```

Calling with `split="validation"` must raise in formal mode.

### `train_acpr_flowcal_v2.py`

Required classes/functions:

```python
class StageController:
    def stage_for_epoch(...)
    def apply(...)
    def validate_trainable_manifest(...)

class StageAwareScheduler: ...
class TestBestSelector: ...
class CheckpointMigratorV1ToV2: ...

def build_optimizer_groups(...)
def train_one_epoch(...)
def evaluate_after_epoch(...)
def save_checkpoint_atomic(...)
def load_resume_exact(...)
def run_formal_suite(...)
```

Main CLI must execute the suite, not just write it.

### `eval_acpr_flowcal_v2.py`

Required:

```text
ADAPT-aligned action/explanation generation metrics
continuous speed/course metrics
traffic-state middle-output JSONL
lightweight per-epoch relevance audit
no model-state mutation
```

### `run_acpr_flowcal_v2_preflight.py`

Must orchestrate gates B–J and stop on first blocker while preserving reports.

### `audit_acpr_flowcal_v2.py`

Must produce all required audit files and only write review pass when all are valid for current SHA.

Static string scans alone are insufficient.

### `probe_acpr_flowcal_v2_memory.py`

Main and SCST candidate probes; records allocated/reserved/peak and finite/step status.

### `supervise_acpr_flowcal_v2_foreground.py`

Required:

```text
pass/SHA verification
attached subprocess
stdout/stderr streaming
heartbeat thread that terminates with parent
status JSON
epoch artifact checks
transient retry
OOM fallback
latest resume
explicit user stop sentinel
full-suite completion check
```

It may use a short-lived heartbeat thread, but no detached child or daemon that continues after parent exit.

## E. Explainability files

### `acpr_flowcal_v2_faithfulness.py`

Required:

```text
normalized text/speed/course effects
conditional subset assignment
direction consistency
paired bootstrap
paired permutation
evidence-vs-random comparison
```

### `acpr_flowcal_v2_renderer.py`

Must render real tensors and generated text into PNG + JSON.

### `acpr_flowcal_v2_atlas.py`

Must create JSON index + standalone HTML grouped by traffic state, action, and failure type.

## F. Scripts

### PowerShell launcher

`FATE_X_acpr_flowcal_v2_foreground.ps1`:

- verify current path/branch;
- verify review pass/current local/current remote SHA;
- invoke Python supervisor directly;
- do not use PowerShell background primitives.

### Bash launcher

Equivalent WSL/Linux foreground launcher.

## G. Implementation manifest

Create `docs/runbooks/ACPR_FlowCal_V2_Implementation_Manifest.json` containing:

```text
formal entrypoints
all formal modules
all public symbols
config consumer map
tests mapped to plan sections
forbidden import list
expected tensor contracts
expected stage trainables
review pass path
```

The audit must compare this manifest against runtime discovery.

## H. Required implementation order

1. Git safety snapshot and install plan/config/skill.
2. Typed config/types and failing contract tests.
3. ADAPT video/text/motion exact baseline wrappers.
4. Local transport and map warping.
5. Transported predicate tracker.
6. Lane-flow field.
7. Axis-aware flow composer.
8. Contextual reason target and PU.
9. Semantic memory and firewall.
10. SECA and control readers.
11. Prefix future and HardPair.
12. Losses, stage controller, optimizer/scheduler.
13. V1 checkpoint migration.
14. SCST.
15. Test-only evaluator and best selector.
16. Interventions/relevance.
17. Canvas/atlas.
18. Preflight/audit/supervisor.
19. Independent review loop.
20. Formal foreground execution.

No later item may be declared complete while an earlier contract test is failing.
