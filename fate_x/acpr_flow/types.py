from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class ACPRFlowBatch:
    input_ids: Tensor | None
    attention_mask: Tensor | None
    token_type_ids: Tensor | None
    frames: Tensor
    masked_pos: Tensor | None = None
    masked_ids: Tensor | None = None
    car_info: Tensor | None = None
    sample_ids: list[str] = field(default_factory=list)
    raw_actions: list[str] = field(default_factory=list)
    raw_justifications: list[str] = field(default_factory=list)


@dataclass
class ACPRFlowBundle:
    fine_grid: Tensor
    coarse_grid: Tensor
    fused_grid: Tensor
    camera_shift: Tensor
    local_transport_probs: Tensor
    transport_dustbin: Tensor
    predicate_attention: Tensor
    predicate_tokens_temporal: Tensor
    predicate_logits_temporal: Tensor
    predicate_probs_temporal: Tensor
    predicate_confidence: Tensor
    predicate_relative_motion: Tensor
    predicate_descriptor: Tensor
    flow_tokens: Tensor
    flow_logits: Tensor
    flow_probs: Tensor
    flow_to_predicate_attention: Tensor
    flow_evidence_maps: Tensor
    local_reason_memory: Tensor
    flow_reason_memory: Tensor
    null_reason_memory: Tensor
    reason_memory: Tensor
    reason_memory_mask: Tensor
    global_reason_state: Tensor
    token_reason_attention: Tensor | None = None
    token_delta: Tensor | None = None
    control_reason_attention: Tensor | None = None
    control_delta: Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ACPRFlowTrainOutput:
    action_text_loss: Tensor
    explanation_text_loss: Tensor
    text_loss_total: Tensor
    control_loss: Tensor
    control_base_prediction: Tensor
    control_final_prediction: Tensor
    baseline_masked_logits: Tensor
    enhanced_masked_logits: Tensor
    auxiliary_loss: Tensor
    total_loss: Tensor
    loss_components: dict[str, Tensor]
    bundle: ACPRFlowBundle
