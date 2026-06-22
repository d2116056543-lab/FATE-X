from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import AxisAwareFlowOutput, LaneFlowFieldOutput, PredicateTrajectory, SemanticReasonMemory


def _memory_names(predicates: PredicateTrajectory, flow_state: AxisAwareFlowOutput) -> Tuple[str, ...]:
    return tuple(predicates.names) + tuple(flow_state.semantic_names) + ("lane_left", "lane_center", "lane_right", "axis_longitudinal", "axis_lateral", "direction_left", "direction_neutral", "direction_right", "null")


class SemanticReasonMemoryBuilder(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 768):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)

    def forward(self, predicates: PredicateTrajectory, lane_flow: LaneFlowFieldOutput, flow_state: AxisAwareFlowOutput) -> SemanticReasonMemory:
        b = predicates.tokens.shape[0]
        pred = predicates.tokens.mean(1)
        parts = [pred, flow_state.semantic_tokens, lane_flow.descriptor, flow_state.axis_tokens, flow_state.direction_tokens]
        raw = torch.cat(parts, dim=1)
        if raw.shape[1] < 53:
            raw = F.pad(raw, (0, 0, 0, 53 - raw.shape[1]))
        raw = raw[:, :53]
        null = raw.new_zeros(b, 1, raw.shape[-1])
        raw = torch.cat([raw, null], dim=1)
        values = self.project(raw)
        mask = torch.ones(b, 54, dtype=torch.bool, device=values.device)
        confidence = torch.ones(b, 54, device=values.device)
        confidence[:, -1] = 0.25
        type_ids = torch.cat([
            torch.zeros(32), torch.ones(13), torch.full((3,), 2), torch.full((2,), 3), torch.full((3,), 4), torch.full((1,), 5)
        ]).long().to(values.device)
        axis_ids = torch.zeros(54, dtype=torch.long, device=values.device)
        axis_ids[32:45] = 3
        axis_ids[48] = 1
        axis_ids[49] = 2
        evidence = values.new_zeros(b, predicates.attention.shape[1], 54, predicates.attention.shape[-2], predicates.attention.shape[-1])
        names = _memory_names(predicates, flow_state)
        semantic_state = (values * confidence.unsqueeze(-1)).sum(1) / confidence.sum(1, keepdim=True).clamp_min(1e-6)
        return SemanticReasonMemory(values, mask, confidence, names, type_ids, axis_ids, evidence, [{"source": n} for n in names], semantic_state)


def longitudinal_memory_mask(memory: SemanticReasonMemory) -> Tensor:
    return (memory.axis_ids == 1) | (memory.axis_ids == 3)


def lateral_memory_mask(memory: SemanticReasonMemory) -> Tensor:
    return (memory.axis_ids == 2) | (memory.axis_ids == 3)
