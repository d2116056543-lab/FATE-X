from __future__ import annotations

import torch
from torch import Tensor, nn

from .types import ExactDecisionLedger


class ExactDecisionLedgerHead(nn.Module):
    def __init__(self, dim: int, factor_count: int = 13, signal_names: tuple[str, str] = ("course", "speed")):
        super().__init__()
        self.signal_names = signal_names
        self.course_reader = nn.Linear(dim, 1)
        self.speed_reader = nn.Linear(dim, 1)
        self.course_benefit = nn.Linear(dim, 1)
        self.speed_benefit = nn.Linear(dim, 1)

    def forward(self, global_prediction_normalized: Tensor, factor_tokens: Tensor) -> ExactDecisionLedger:
        raw = torch.cat([self.course_reader(factor_tokens), self.speed_reader(factor_tokens)], dim=-1)
        gate = torch.sigmoid(
            torch.cat([self.course_benefit(factor_tokens), self.speed_benefit(factor_tokens)], dim=-1)
        )
        gated = raw * gate
        final = global_prediction_normalized.float() + gated.float().sum(dim=2)
        speed_attention = torch.softmax(raw[..., 1].abs(), dim=-1)
        course_attention = torch.softmax(raw[..., 0].abs(), dim=-1)
        return ExactDecisionLedger(
            signal_names=self.signal_names,
            global_prediction_normalized=global_prediction_normalized.float(),
            raw_factor_contributions_normalized=raw,
            benefit_gate=gate,
            gated_factor_contributions_normalized=gated.float(),
            final_prediction_normalized=final,
            global_prediction_raw=global_prediction_normalized.float(),
            gated_factor_contributions_raw=gated.float(),
            final_prediction_raw=final,
            speed_factor_attention=speed_attention,
            course_factor_attention=course_attention,
            benefit_target=None,
        )
