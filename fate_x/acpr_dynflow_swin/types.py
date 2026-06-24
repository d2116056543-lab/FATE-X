from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass
class DynFlowSwinBatch:
    frames: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    masked_pos: Tensor
    masked_ids: Tensor
    control_target: Tensor | None
    sample_ids: list[str]
    raw_actions: list[str]
    raw_justifications: list[str]


@dataclass
class SwinBackboneOutput:
    predicate_grid: Tensor
    final_grid: Tensor
    temporal_global: Tensor
    dense_final_tokens: Tensor
    forward_count: int


@dataclass
class DynamicPredicateField:
    names: tuple[str, ...]
    query_states: Tensor
    logits: Tensor
    probabilities: Tensor
    tokens: Tensor
    evidence_maps: Tensor
    confidence: Tensor
    centroid: Tensor
    relative_motion: Tensor
    corridor_mass: Tensor
    transfer_gate: Tensor


@dataclass
class SemanticTokenConsolidation:
    slot_names: tuple[str, ...]
    assignment: Tensor
    token_mass: Tensor
    tokens: Tensor
    source_provenance: Tensor
    conservation_error: Tensor


@dataclass
class TrafficStateOutput:
    factor_names: tuple[str, ...]
    factor_tokens_native: Tensor
    factor_logits: Tensor
    factor_probs: Tensor
    lateral_bias: Tensor
    pattern_logits: Tensor
    pattern_probs: Tensor
    factor_to_predicate: Tensor
    factor_to_corridor: Tensor
    evidence_maps: Tensor
    lag_weights: Tensor
    lag_aligned_tokens: Tensor
    lineage: list[dict[str, Any]]


@dataclass
class MotionTransformerOutput:
    query_hidden: Tensor
    global_prediction_normalized: Tensor
    source_attention: Tensor | None


@dataclass
class ExactDecisionLedger:
    signal_names: tuple[str, ...]
    global_prediction_normalized: Tensor
    raw_factor_contributions_normalized: Tensor
    benefit_gate: Tensor
    gated_factor_contributions_normalized: Tensor
    final_prediction_normalized: Tensor
    global_prediction_raw: Tensor
    gated_factor_contributions_raw: Tensor
    final_prediction_raw: Tensor
    speed_factor_attention: Tensor
    course_factor_attention: Tensor
    benefit_target: Tensor | None


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
