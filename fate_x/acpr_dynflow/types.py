from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class DynFlowBatch:
    frames: Tensor
    input_ids: Tensor | None
    attention_mask: Tensor | None
    token_type_ids: Tensor | None
    masked_pos: Tensor | None
    masked_ids: Tensor | None
    control_target: Tensor | None
    sample_ids: list[str]
    raw_actions: list[str]
    raw_justifications: list[str]


@dataclass
class DynFlowBackboneOutput:
    local_grid: Tensor
    coarse_grid: Tensor
    global_sequence: Tensor
    text_visual_tokens: Tensor
    forward_count: int


@dataclass
class DynamicPredicateField:
    names: tuple[str, ...]
    logits: Tensor
    probabilities: Tensor
    tokens: Tensor
    evidence_maps: Tensor
    confidence: Tensor
    centroid: Tensor
    relative_centroid_motion: Tensor
    lane_mass: Tensor
    query_states: Tensor


@dataclass
class PredicateCovariates:
    raw_covariates: Tensor
    homogenized: Tensor
    multiscale: dict[str, Tensor]
    pattern_logits: Tensor
    pattern_probs: Tensor
    pattern_names: tuple[str, ...]


@dataclass
class TrafficFlowState:
    factor_names: tuple[str, ...]
    factor_tokens: Tensor
    factor_logits: Tensor
    factor_probs: Tensor
    lateral_bias: Tensor
    factor_to_predicate: Tensor
    evidence_maps: Tensor
    response_lag_weights: Tensor
    lag_aligned_tokens: Tensor
    lineage: list[dict[str, Any]]


@dataclass
class DecisionLedger:
    signal_names: tuple[str, ...]
    global_prediction_normalized: Tensor
    factor_contributions_normalized: Tensor
    final_prediction_normalized: Tensor
    global_prediction_raw: Tensor
    factor_contributions_raw: Tensor
    final_prediction_raw: Tensor
    speed_factor_attention: Tensor
    course_factor_attention: Tensor


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

