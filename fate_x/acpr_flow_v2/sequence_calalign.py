from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class SequenceCalAlignV2Scales:
    alpha_action: float = 0.0
    alpha_explanation: float = 0.0
    alpha_speed: float = 0.0
    alpha_course: float = 0.0
    temperature_action: float = 1.0
    temperature_explanation: float = 1.0


class SequenceCalAlignV2:
    def __init__(self):
        self.scales = SequenceCalAlignV2Scales()

    def fit_text(self, baseline_logits: Tensor, enhanced_logits: Tensor, labels: Tensor, branch: str = "explanation") -> float:
        alphas = [0.0, 0.05, 0.1, 0.2]
        best = min(alphas, key=lambda a: F.cross_entropy(baseline_logits + a * (enhanced_logits - baseline_logits), labels).item())
        if branch == "action":
            self.scales.alpha_action = best
        else:
            self.scales.alpha_explanation = best
        return best

    def fit_control(self, base: Tensor, enhanced: Tensor, target: Tensor, branch: str = "speed") -> float:
        alphas = [0.0, 0.05, 0.1, 0.2]
        best = min(alphas, key=lambda a: F.mse_loss(base + a * (enhanced - base), target).item())
        if branch == "course":
            self.scales.alpha_course = best
        else:
            self.scales.alpha_speed = best
        return best

    def apply_text(self, baseline_logits: Tensor, enhanced_logits: Tensor, branch: str = "explanation") -> Tensor:
        alpha = self.scales.alpha_action if branch == "action" else self.scales.alpha_explanation
        temp = self.scales.temperature_action if branch == "action" else self.scales.temperature_explanation
        return (baseline_logits + alpha * (enhanced_logits - baseline_logits)) / temp

    def apply_control(self, base: Tensor, enhanced: Tensor, branch: str = "speed") -> Tensor:
        alpha = self.scales.alpha_course if branch == "course" else self.scales.alpha_speed
        return base + alpha * (enhanced - base)

    def fit(self, *args, **kwargs) -> "SequenceCalAlignV2":
        return self

    def transform(self, base: Tensor, enhanced: Tensor, branch: str = "speed") -> Tensor:
        return self.apply_control(base, enhanced, branch=branch)


SequenceCalAlign = SequenceCalAlignV2
