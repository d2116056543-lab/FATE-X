from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class PUTargetBatch:
    targets: Tensor
    known_mask: Tensor
    positive_mask: Tensor
    unknown_weight: float


class FreeTextPUTargetBuilderV2:
    def __init__(self, names: Sequence[str] = ("slow", "stop", "turn", "vehicle"), unknown_weight: float = 0.005):
        self.names = tuple(names)
        self.unknown_weight = unknown_weight

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "FreeTextPUTargetBuilderV2":
        return cls()

    def build(self, actions: List[str], justifications: List[str], epoch: int = 0) -> PUTargetBatch:
        texts = [f"{a} {j}".lower() for a, j in zip(actions, justifications)]
        rows = []
        for text in texts:
            rows.append([1.0 if name in text else 0.0 for name in self.names])
        targets = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, len(self.names))
        known = targets > 0
        return PUTargetBatch(targets=targets, known_mask=known, positive_mask=targets.bool(), unknown_weight=self.unknown_weight)


def positive_unlabeled_loss_v2(logits: Tensor, targets: Tensor, known_mask: Optional[Tensor] = None, unknown_weight: float = 0.005) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    if known_mask is None:
        weights = torch.where(targets > 0, torch.ones_like(targets), torch.full_like(targets, unknown_weight))
    else:
        weights = torch.where(known_mask.bool(), torch.ones_like(targets), torch.full_like(targets, unknown_weight))
    return (bce * weights).sum() / weights.sum().clamp_min(1.0)


positive_unlabeled_bce = positive_unlabeled_loss_v2
