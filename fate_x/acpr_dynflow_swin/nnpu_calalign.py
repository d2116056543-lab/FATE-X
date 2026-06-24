from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class PULabels:
    positive: Tensor
    reliable_negative: Tensor
    unlabeled: Tensor


def nnpu_loss(logits: Tensor, labels: PULabels, prior: Tensor) -> Tensor:
    positive = labels.positive.to(logits)
    reliable_negative = labels.reliable_negative.to(logits)
    unlabeled = labels.unlabeled.to(logits)
    prior = prior.to(logits).flatten()

    def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        return (values * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)

    positive_risk = prior * masked_mean(F.softplus(-logits), positive)
    positive_as_negative = prior * masked_mean(F.softplus(logits), positive)
    unlabeled_negative = masked_mean(F.softplus(logits), unlabeled)
    reliable_negative_risk = masked_mean(F.softplus(logits), reliable_negative)
    nonnegative_unlabeled = torch.clamp(unlabeled_negative - positive_as_negative, min=0.0)
    active = (positive.sum(dim=0) + reliable_negative.sum(dim=0) + unlabeled.sum(dim=0)).gt(0)
    per_predicate = positive_risk + reliable_negative_risk + nonnegative_unlabeled
    if not bool(active.any()):
        return logits.sum() * 0.0
    return per_predicate[active].mean()


class PredicateRuleLabeler:
    def __init__(self, rules: dict[str, dict[str, tuple[str, ...]]]):
        self.rules = rules

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PredicateRuleLabeler":
        from .predicate_ontology import EXACT_32_PREDICATES

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        groups = {
            "positive": payload.get("positive_rules", {}),
            "reliable_negative": payload.get("reliable_negative_rules", {}),
            "exclusion": payload.get("exclusion_rules", {}),
        }
        rules: dict[str, dict[str, tuple[str, ...]]] = {}
        for name in EXACT_32_PREDICATES:
            rules[name] = {
                group: tuple(str(item).lower() for item in values.get(name, []))
                for group, values in groups.items()
            }
        return cls(rules)

    def label(self, texts: Iterable[str], device: torch.device | None = None) -> PULabels:
        normalized = [" ".join(str(text).lower().split()) for text in texts]
        positive = torch.zeros(len(normalized), len(self.rules), dtype=torch.float32, device=device)
        reliable_negative = torch.zeros_like(positive)
        excluded = torch.zeros_like(positive)
        for row, text in enumerate(normalized):
            for column, rule in enumerate(self.rules.values()):
                if any(phrase and phrase in text for phrase in rule["exclusion"]):
                    excluded[row, column] = 1
                    continue
                if any(phrase and phrase in text for phrase in rule["positive"]):
                    positive[row, column] = 1
                elif any(phrase and phrase in text for phrase in rule["reliable_negative"]):
                    reliable_negative[row, column] = 1
        unlabeled = (1.0 - positive - reliable_negative - excluded).clamp(0, 1)
        return PULabels(positive=positive, reliable_negative=reliable_negative, unlabeled=unlabeled)


class OnlineCalAlign(nn.Module):
    def __init__(self, num_predicates: int = 32):
        super().__init__()
        self.register_buffer("prior", torch.full((num_predicates,), 0.1))
        self.register_buffer("temperature", torch.ones(num_predicates))
        self.register_buffer("positive_threshold", torch.full((num_predicates,), 0.7))
        self.register_buffer("negative_threshold", torch.full((num_predicates,), 0.3))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))

    def calibrate(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits / self.temperature.clamp_min(1e-3))

    @torch.no_grad()
    def update(self, logits: Tensor, labels: PULabels, training: bool, momentum: float = 0.95) -> None:
        if not training:
            return
        probs = torch.sigmoid(logits.detach().float())
        positive = labels.positive.to(device=probs.device, dtype=torch.float32)
        negative = labels.reliable_negative.to(device=probs.device, dtype=torch.float32)
        positive_count = positive.sum(dim=0)
        negative_count = negative.sum(dim=0)
        observed_prior = positive.mean(dim=0)
        self.prior.lerp_(observed_prior.clamp(0.01, 0.99), 1.0 - momentum)
        positive_mean = (probs * positive).sum(dim=0) / positive_count.clamp_min(1.0)
        negative_mean = (probs * negative).sum(dim=0) / negative_count.clamp_min(1.0)
        self.positive_threshold.lerp_(
            torch.where(positive_count.gt(0), positive_mean, self.positive_threshold), 1.0 - momentum
        )
        self.negative_threshold.lerp_(
            torch.where(negative_count.gt(0), negative_mean, self.negative_threshold), 1.0 - momentum
        )
        spread = probs.std(dim=0, unbiased=False).clamp(0.25, 2.0)
        self.temperature.lerp_(spread, 1.0 - momentum)
        self.update_count.add_(1)
