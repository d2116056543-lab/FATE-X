from __future__ import annotations

import torch
from torch import Tensor, nn

from .types import DynamicPredicateField, PredicateCovariates


class MesoscopicLaneFlow(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.encoder = nn.Linear(6, dim)

    def forward(self, pred: DynamicPredicateField, cov: PredicateCovariates) -> dict[str, Tensor]:
        occ = (pred.probabilities.unsqueeze(-1) * pred.lane_mass).mean(2)
        mobility = torch.cat([torch.zeros_like(occ[:, :1]), occ[:, 1:] - occ[:, :-1]], dim=1)
        stopped = torch.relu(-mobility)
        queue = occ * (1.0 + stopped)
        gap = 1.0 - occ.clamp(0, 1)
        rel_motion = pred.relative_centroid_motion.norm(dim=-1).mean(2)
        rel_motion = torch.cat([torch.zeros_like(rel_motion[:, :1]), rel_motion], dim=1)
        coherence = 1.0 / (1.0 + rel_motion.unsqueeze(-1).expand_as(occ))
        desc = torch.stack([occ, mobility.abs(), coherence, stopped, queue, gap], dim=-1)
        tokens = self.encoder(desc)
        return {"descriptors": desc, "tokens": tokens, "corridors": torch.tensor([-1.0, 0.0, 1.0], device=occ.device)}

