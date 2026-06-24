from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class PULabels:
    positive: Tensor
    reliable_negative: Tensor
    unlabeled: Tensor


def nnpu_loss(logits: Tensor, labels: PULabels, prior: Tensor) -> Tensor:
    positive_risk = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none") * labels.positive
    negative_risk = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none") * labels.reliable_negative
    unlabeled_risk = F.binary_cross_entropy_with_logits(-logits, torch.ones_like(logits), reduction="none") * labels.unlabeled
    return positive_risk.mean() + torch.clamp(negative_risk.mean() + prior.mean() * unlabeled_risk.mean(), min=0.0)


class OnlineCalAlign(nn.Module):
    def __init__(self, num_predicates: int = 32):
        super().__init__()
        self.register_buffer("prior", torch.full((num_predicates,), 0.1))
        self.register_buffer("temperature", torch.ones(num_predicates))
        self.register_buffer("threshold", torch.full((num_predicates,), 0.5))

    def calibrate(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits / self.temperature.clamp_min(1e-3))
