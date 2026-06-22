from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class FlowCalV2Batch:
    frames: Tensor
    input_ids: Optional[Tensor] = None
    attention_mask: Optional[Tensor] = None
    token_type_ids: Optional[Tensor] = None
    masked_pos: Optional[Tensor] = None
    masked_ids: Optional[Tensor] = None
    car_info: Optional[Tensor] = None
    sample_ids: List[str] = field(default_factory=list)
    raw_actions: List[str] = field(default_factory=list)
    raw_justifications: List[str] = field(default_factory=list)


@dataclass
class VideoBackboneOutput:
    fine_native: Tensor
    coarse_native: Tensor
    fine_aligned: Tensor
    coarse_aligned: Tensor
    fused_grid: Tensor
    dense_tokens_raw: Tensor
    dense_tokens_projected: Tensor
    forward_count: int


@dataclass
class LocalTransportOutput:
    probs: Tensor
    candidate_offsets: Tensor
    expected_displacement: Tensor
    dustbin_prob: Tensor
    common_shift: Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredicateTrajectory:
    names: Tuple[str, ...]
    attention: Tensor
    tokens: Tensor
    presence_logits: Tensor
    presence_probs: Tensor
    confidence: Tensor
    relative_motion: Tensor
    descriptor: Tensor
    descriptor_parts: Dict[str, Tensor] = field(default_factory=dict)


@dataclass
class LaneFlowFieldOutput:
    region_names: Tuple[str, str, str]
    soft_masks: Tensor
    occupancy: Tensor
    relative_motion: Tensor
    motion_coherence: Tensor
    stopped_tendency: Tensor
    queue_pressure: Tensor
    temporal_tokens: Tensor
    descriptor: Tensor


@dataclass
class AxisAwareFlowOutput:
    semantic_names: Tuple[str, ...]
    semantic_tokens: Tensor
    semantic_logits: Tensor
    semantic_probs: Tensor
    semantic_evidence: Tensor
    lane_tokens: Tensor
    axis_tokens: Tensor
    axis_logits: Tensor
    axis_probs: Tensor
    direction_tokens: Tensor
    direction_logits: Tensor
    direction_probs: Tensor
    flow_to_predicate_attention: Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticReasonMemory:
    values: Tensor
    mask: Tensor
    confidence: Tensor
    names: Tuple[str, ...]
    type_ids: Tensor
    axis_ids: Tensor
    evidence_maps: Tensor
    lineage: List[Dict[str, Any]]
    semantic_state: Tensor


@dataclass
class GeneratedSequence:
    token_ids: Tensor
    logprobs: Tensor
    texts: List[str] = field(default_factory=list)


@dataclass
class InterventionSpecV2:
    kind: str
    target: Optional[str] = None
    strength: float = 1.0
    seed: int = 0


@dataclass
class FlowCalV2Bundle:
    video: Optional[VideoBackboneOutput] = None
    local_transport: Optional[LocalTransportOutput] = None
    predicates: Optional[PredicateTrajectory] = None
    lane_flow: Optional[LaneFlowFieldOutput] = None
    flow_state: Optional[AxisAwareFlowOutput] = None
    reason_memory: Optional[SemanticReasonMemory] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def local_transport_probs(self) -> Tensor:
        if self.local_transport is None:
            raise AttributeError("local_transport is not populated")
        return self.local_transport.probs


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
    control_base_prediction: Tensor
    control_final_prediction: Tensor
    control_hidden: Tensor
    loss_components: Dict[str, Tensor]
    gradient_diagnostics: Dict[str, Tensor]
    bundle: FlowCalV2Bundle

    @property
    def text_logits(self) -> Tensor:
        return self.enhanced_masked_logits

    @property
    def control_pred(self) -> Tensor:
        return self.control_final_prediction


# Backward-compatible alias used by the earlier smoke tests.
FlowCalV2Output = FlowCalV2TrainOutput
