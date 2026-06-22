from __future__ import annotations

import torch
from torch import Tensor, nn

from .predicate_ontology import PATTERN_NAMES
from .types import DynamicPredicateField, PredicateCovariates


class PredicateCovariateHomogenizer(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(10, out_dim), nn.ReLU(), nn.LayerNorm(out_dim))

    def forward(self, pred: DynamicPredicateField) -> PredicateCovariates:
        b, t, k = pred.probabilities.shape
        zero_motion = torch.zeros(b, 1, k, 2, device=pred.probabilities.device, dtype=pred.probabilities.dtype)
        rel_motion = torch.cat([zero_motion, pred.relative_centroid_motion], dim=1)
        presence_delta = torch.cat([torch.zeros_like(pred.probabilities[:, :1]), pred.probabilities[:, 1:] - pred.probabilities[:, :-1]], dim=1).unsqueeze(-1)
        raw = torch.cat([
            pred.probabilities.unsqueeze(-1),
            presence_delta,
            pred.centroid,
            rel_motion,
            pred.lane_mass,
            pred.confidence.unsqueeze(-1),
        ], dim=-1)
        homog = self.proj(raw)
        return PredicateCovariates(raw_covariates=raw, homogenized=homog, multiscale={}, pattern_logits=torch.empty(0, device=homog.device), pattern_probs=torch.empty(0, device=homog.device), pattern_names=PATTERN_NAMES)

