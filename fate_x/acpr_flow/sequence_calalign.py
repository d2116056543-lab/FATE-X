from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class SequenceCalAlignScales:
    alpha_action: float = 0.0
    alpha_explanation: float = 0.0
    alpha_speed: float = 0.0
    alpha_course: float = 0.0
    temperature_action: float = 1.0
    temperature_explanation: float = 1.0


class SequenceCalAlign:
    def __init__(self, train_calib_ids: list[str]) -> None:
        self.train_calib_ids = set(train_calib_ids)
        self.scales = SequenceCalAlignScales()
        self.fit_uses_test = False

    def fit(self, sample_ids: list[str], base_logits: Tensor, enhanced_logits: Tensor, targets: Tensor,
            alpha_grid: list[float] | None = None, temperature_grid: list[float] | None = None) -> SequenceCalAlignScales:
        if not set(sample_ids).issubset(self.train_calib_ids):
            raise ValueError("Sequence-CalAlign fit received non train-calib sample ids")
        alpha_grid = alpha_grid or [0.0, 0.05, 0.10, 0.15]
        temperature_grid = temperature_grid or [0.85, 1.0, 1.15]
        best = (float("inf"), 0.0, 1.0)
        for a in alpha_grid:
            logits = base_logits.detach() + a * (enhanced_logits.detach() - base_logits.detach())
            for temp in temperature_grid:
                loss = F.cross_entropy(logits / temp, targets).item()
                if loss < best[0]:
                    best = (loss, a, temp)
        self.scales = SequenceCalAlignScales(alpha_action=best[1], temperature_action=best[2])
        return self.scales

    def apply_action(self, base_logits: Tensor, enhanced_logits: Tensor) -> Tensor:
        return base_logits + self.scales.alpha_action * (enhanced_logits - base_logits) / self.scales.temperature_action
