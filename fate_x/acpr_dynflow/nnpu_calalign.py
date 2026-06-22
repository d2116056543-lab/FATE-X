from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class PULabels:
    positive: Tensor
    reliable_negative: Tensor
    unlabeled: Tensor


def phrase_labels(texts: list[str], rules: Mapping[str, Mapping[str, list[str]]], names: tuple[str, ...], device: torch.device) -> PULabels:
    pos = torch.zeros(len(texts), len(names), device=device)
    neg = torch.zeros_like(pos)
    for i, text in enumerate(t.lower() for t in texts):
        for k, name in enumerate(names):
            rule = rules.get(name, rules.get("default", {}))
            if any(ex and ex in text for ex in rule.get("exclusion", [])):
                continue
            if any(p and p in text for p in rule.get("positive", [])):
                pos[i, k] = 1.0
            if any(c and c in text for c in rule.get("contradiction", [])):
                neg[i, k] = 1.0
    unlabeled = 1.0 - torch.clamp(pos + neg, 0, 1)
    return PULabels(pos, neg, unlabeled)


def nnpu_risk(logits: Tensor, labels: PULabels, prior: Tensor | float = 0.1) -> Tensor:
    prior_t = torch.as_tensor(prior, device=logits.device, dtype=logits.dtype)
    probs = logits
    while prior_t.ndim < probs.ndim:
        prior_t = prior_t.unsqueeze(0)
    pos_loss = F.binary_cross_entropy_with_logits(probs, torch.ones_like(probs), reduction="none")
    neg_loss = F.binary_cross_entropy_with_logits(probs, torch.zeros_like(probs), reduction="none")
    p_mask = labels.positive
    rn_mask = labels.reliable_negative
    while p_mask.ndim < probs.ndim:
        p_mask = p_mask.unsqueeze(1)
        rn_mask = rn_mask.unsqueeze(1)
    pos_term = (pos_loss * p_mask).sum() / p_mask.sum().clamp_min(1.0)
    neg_term = (neg_loss * rn_mask).sum() / rn_mask.sum().clamp_min(1.0)
    return prior_t.mean() * pos_term + torch.relu(neg_term - prior_t.mean() * pos_term)


class OnlineCalAlign(nn.Module):
    def __init__(self, num_predicates: int):
        super().__init__()
        self.register_buffer("prior", torch.full((num_predicates,), 0.1))
        self.threshold = nn.Parameter(torch.zeros(num_predicates), requires_grad=False)
        self.temperature = nn.Parameter(torch.ones(num_predicates), requires_grad=False)

    def calibrate(self, logits: Tensor) -> Tensor:
        return (logits - self.threshold) / self.temperature.clamp_min(1e-3)

