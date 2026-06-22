from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import LocalTransportOutput, PredicateTrajectory

PREDICATE_NAMES = tuple(f"predicate_{i:02d}" for i in range(32))


class TransportedNamedPredicateTracker(nn.Module):
    def __init__(self, dim: int = 256, num_predicates: int = 32):
        super().__init__()
        self.num_predicates = num_predicates
        self.query = nn.Parameter(torch.randn(num_predicates, dim) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, fused_grid: Tensor, transport: LocalTransportOutput) -> PredicateTrajectory:
        b, t, h, w, d = fused_grid.shape
        q = F.normalize(self.query, dim=-1)
        x = F.normalize(fused_grid, dim=-1)
        logits = torch.einsum("bthwd,pd->btphw", x, q) / self.temperature.clamp_min(0.2)
        attention = torch.softmax(logits.flatten(-2), dim=-1).view(b, t, self.num_predicates, h, w)
        tokens = torch.einsum("btphw,bthwd->btpd", attention, fused_grid)
        presence_logits = logits.amax(dim=(-1, -2))
        presence_probs = torch.sigmoid(presence_logits)
        concentration = attention.flatten(-2).amax(-1)
        confidence = (presence_probs * concentration).clamp(0, 1)
        if t > 1:
            rel = transport.expected_displacement.mean(dim=(2, 3)).unsqueeze(2).expand(b, t - 1, self.num_predicates, 2)
        else:
            rel = fused_grid.new_zeros(b, 0, self.num_predicates, 2)
        now = tokens[:, -1].mean(1)
        hist = (tokens * confidence.unsqueeze(-1)).sum((1, 2)) / confidence.sum((1, 2), keepdim=False).unsqueeze(-1).clamp_min(1e-6)
        descriptor = torch.cat([now, hist], dim=-1)
        descriptor = descriptor[..., :d] if descriptor.shape[-1] >= d else F.pad(descriptor, (0, d - descriptor.shape[-1]))
        return PredicateTrajectory(
            names=PREDICATE_NAMES[: self.num_predicates],
            attention=attention,
            tokens=tokens,
            presence_logits=presence_logits,
            presence_probs=presence_probs,
            confidence=confidence,
            relative_motion=rel,
            descriptor=descriptor,
            descriptor_parts={"now": now, "history": hist, "presence_rate": presence_probs.mean(1)},
        )


TemporalPredicateTracker = TransportedNamedPredicateTracker
