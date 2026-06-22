from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import AxisAwareFlowOutput, LaneFlowFieldOutput, PredicateTrajectory

SEMANTIC_NAMES = (
    "clear_open_flow", "stable_following", "dense_following", "queue_congestion",
    "forming", "stable", "releasing", "oscillating",
    "traffic_signal", "lead_vehicle_group", "merge_lane_constraint",
    "turn_intersection", "vulnerable_obstacle_conflict",
)


def derive_axis_direction_targets(control_targets: Tensor, signal_names: Sequence[str], control_stats: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
    idx = {name: i for i, name in enumerate(signal_names)}
    if control_targets.shape[-1] == len(signal_names):
        speed = control_targets[..., idx.get("speed", min(1, control_targets.shape[-1] - 1))]
        course = control_targets[..., idx.get("course", 0)]
    else:
        speed = control_targets[:, idx.get("speed", min(1, control_targets.shape[1] - 1))]
        course = control_targets[:, idx.get("course", 0)]
    speed_delta = speed[..., -1] - speed[..., 0]
    course_delta = course[..., -1] - course[..., 0]
    course_level = course.mean(dim=-1)
    signed_course = torch.where(course_delta.abs() > 0.1, course_delta, course_level)
    axis = torch.stack([speed_delta.abs() > 0.1, course_delta.abs() > 0.1], dim=-1).float()
    direction = torch.zeros(control_targets.shape[0], 3, device=control_targets.device)
    direction[:, 0] = (signed_course < -0.1).float()
    direction[:, 1] = (signed_course.abs() <= 0.1).float()
    direction[:, 2] = (signed_course > 0.1).float()
    return {
        "axis_targets": axis,
        "direction_targets": direction,
        "longitudinal": axis[:, 0],
        "lateral": axis[:, 1],
        "direction": direction,
    }


class AxisAwareFlowComposer(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.semantic_names = SEMANTIC_NAMES
        self.semantic = nn.Linear(dim, len(SEMANTIC_NAMES))
        self.axis = nn.Linear(dim, 2)
        self.direction = nn.Linear(dim, 3)
        self.semantic_embed = nn.Embedding(len(SEMANTIC_NAMES), dim)
        self.axis_embed = nn.Embedding(2, dim)
        self.direction_embed = nn.Embedding(3, dim)

    def forward(self, predicates: PredicateTrajectory, lane_flow: LaneFlowFieldOutput) -> AxisAwareFlowOutput:
        base = lane_flow.descriptor.mean(1)
        semantic_logits = self.semantic(base)
        semantic_probs = torch.sigmoid(semantic_logits)
        axis_logits = self.axis(base)
        axis_probs = torch.sigmoid(axis_logits)
        direction_logits = self.direction(base)
        direction_probs = torch.softmax(direction_logits, dim=-1)
        b = base.shape[0]
        semantic_tokens = self.semantic_embed.weight.unsqueeze(0).expand(b, -1, -1) * semantic_probs.unsqueeze(-1)
        axis_tokens = self.axis_embed.weight.unsqueeze(0).expand(b, -1, -1) * axis_probs.unsqueeze(-1)
        direction_tokens = self.direction_embed.weight.unsqueeze(0).expand(b, -1, -1) * direction_probs.unsqueeze(-1)
        evidence = predicates.attention[:, :, : len(SEMANTIC_NAMES)].mean(2, keepdim=False)
        evidence = evidence.unsqueeze(2).expand(-1, -1, len(SEMANTIC_NAMES), -1, -1)
        attn = torch.softmax(torch.einsum("bsd,bpd->bsp", semantic_tokens, predicates.tokens.mean(1)), dim=-1)
        return AxisAwareFlowOutput(
            semantic_names=SEMANTIC_NAMES,
            semantic_tokens=semantic_tokens,
            semantic_logits=semantic_logits,
            semantic_probs=semantic_probs,
            semantic_evidence=evidence,
            lane_tokens=lane_flow.descriptor,
            axis_tokens=axis_tokens,
            axis_logits=axis_logits,
            axis_probs=axis_probs,
            direction_tokens=direction_tokens,
            direction_logits=direction_logits,
            direction_probs=direction_probs,
            flow_to_predicate_attention=attn,
            diagnostics={"semantic_mean": semantic_probs.mean().detach()},
        )
