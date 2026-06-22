from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .semantic_reason_memory import lateral_memory_mask, longitudinal_memory_mask


@dataclass
class AxisControlOutput:
    final_prediction: Tensor
    residual: Tensor
    speed_attention: Tensor
    course_attention: Tensor
    diagnostics: Dict[str, Tensor]


class AxisAwareReasonControlAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 768, max_residual_std_fraction: float = 0.15):
        super().__init__()
        self.speed_reader = nn.Linear(hidden_dim, 1)
        self.course_reader = nn.Linear(hidden_dim, 1)
        self.max_residual = max_residual_std_fraction

    def forward(self, base_prediction: Tensor, control_hidden: Tensor, memory: Any, control_stats: Optional[Dict[str, Tensor]] = None) -> AxisControlOutput:
        long_mask = longitudinal_memory_mask(memory)
        lat_mask = lateral_memory_mask(memory)
        speed_mem = memory.values.masked_fill(~long_mask.view(1, -1, 1), 0).sum(1) / long_mask.float().sum().clamp_min(1.0)
        course_mem = memory.values.masked_fill(~lat_mask.view(1, -1, 1), 0).sum(1) / lat_mask.float().sum().clamp_min(1.0)
        speed_delta = torch.tanh(self.speed_reader(speed_mem)).unsqueeze(1).expand(-1, base_prediction.shape[1], -1)
        course_delta = torch.tanh(self.course_reader(course_mem)).unsqueeze(1).expand(-1, base_prediction.shape[1], -1)
        residual = torch.cat([course_delta, speed_delta], dim=-1) * self.max_residual
        return AxisControlOutput(base_prediction + residual, residual, long_mask.float(), lat_mask.float(), {"residual_norm": residual.norm(dim=-1).mean().detach()})


AxisAwareControlAdapter = AxisAwareReasonControlAdapter
