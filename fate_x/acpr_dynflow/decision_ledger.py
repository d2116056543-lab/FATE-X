from __future__ import annotations

import torch
from torch import Tensor, nn

from .signal_codec import BDDSignalCodec
from .types import DecisionLedger, TrafficFlowState


class DecisionLedgerHead(nn.Module):
    def __init__(self, dim: int = 256, signal_names: tuple[str, ...] = ("course", "speed")):
        super().__init__()
        self.signal_names = signal_names
        self.speed_reader = nn.Linear(dim, 1)
        self.course_reader = nn.Linear(dim, 1)

    def forward(self, global_norm: Tensor, flow: TrafficFlowState, codec: BDDSignalCodec) -> DecisionLedger:
        course = self.course_reader(flow.lag_aligned_tokens)
        speed = self.speed_reader(flow.lag_aligned_tokens)
        lateral = flow.lateral_bias.squeeze(-1)
        if lateral.shape[1] != course.shape[1]:
            lateral = torch.nn.functional.interpolate(lateral.unsqueeze(1), size=course.shape[1], mode="linear", align_corners=False).squeeze(1)
        # Signed lateral traffic bias must directly affect course contributions.
        course = course + lateral.unsqueeze(-1).unsqueeze(2)
        contrib = torch.cat([course, speed], dim=-1) / max(flow.lag_aligned_tokens.shape[2], 1)
        if global_norm.shape[1] != contrib.shape[1]:
            global_norm = torch.nn.functional.interpolate(global_norm.permute(0, 2, 1), size=contrib.shape[1], mode="linear", align_corners=False).permute(0, 2, 1)
        final = global_norm + contrib.sum(dim=2)
        global_raw = codec.decode(global_norm)
        mean, std = codec._stats(global_norm.device)
        contrib_raw = contrib * std
        final_raw = global_raw + contrib_raw.sum(dim=2)
        speed_attn = torch.softmax(speed.squeeze(-1).abs(), dim=-1)
        course_attn = torch.softmax(course.squeeze(-1).abs(), dim=-1)
        return DecisionLedger(
            signal_names=self.signal_names,
            global_prediction_normalized=global_norm,
            factor_contributions_normalized=contrib,
            final_prediction_normalized=final,
            global_prediction_raw=global_raw,
            factor_contributions_raw=contrib_raw,
            final_prediction_raw=final_raw,
            speed_factor_attention=speed_attn,
            course_factor_attention=course_attn,
        )

