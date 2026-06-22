from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .predicate_ontology import PATTERN_NAMES
from .types import PredicateCovariates


class MultiScalePatternRouter(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.mix1 = nn.Linear(dim, dim)
        self.mix2 = nn.Linear(dim, dim)
        self.mix4 = nn.Linear(dim, dim)
        self.out = nn.Linear(dim * 3, len(PATTERN_NAMES))

    def forward(self, cov: PredicateCovariates) -> PredicateCovariates:
        x = cov.homogenized
        x1 = self.mix1(x)
        x2 = self.mix2(F.avg_pool1d(x.permute(0, 2, 3, 1).reshape(-1, x.shape[-1], x.shape[1]), 2, stride=1, padding=1)[..., : x.shape[1]].reshape(x.shape[0], x.shape[2], x.shape[-1], x.shape[1]).permute(0, 3, 1, 2))
        x4 = self.mix4(F.avg_pool1d(x.permute(0, 2, 3, 1).reshape(-1, x.shape[-1], x.shape[1]), 4, stride=1, padding=2)[..., : x.shape[1]].reshape(x.shape[0], x.shape[2], x.shape[-1], x.shape[1]).permute(0, 3, 1, 2))
        logits = self.out(torch.cat([x1, x2, x4], dim=-1))
        probs = torch.softmax(logits, dim=-1)
        cov.multiscale.update({"scale1": x1, "scale2": x2, "scale4": x4})
        cov.pattern_logits = logits
        cov.pattern_probs = probs
        cov.pattern_names = PATTERN_NAMES
        return cov

