from __future__ import annotations

import torch
from torch import Tensor, nn

from .types import TrafficFlowState


class ResponseLagAligner(nn.Module):
    def __init__(self, dim: int = 256, lags: tuple[int, ...] = (0, 1, 2, 3)):
        super().__init__()
        self.lags = lags
        self.score = nn.Linear(dim, len(lags))

    def forward(self, flow: TrafficFlowState, target_steps: int = 32) -> TrafficFlowState:
        tok = flow.factor_tokens
        if tok.shape[1] < target_steps:
            tok32 = tok.repeat_interleave((target_steps + tok.shape[1] - 1) // tok.shape[1], dim=1)[:, :target_steps]
        else:
            tok32 = tok[:, :target_steps]
        weights = torch.softmax(self.score(tok32), dim=-1)
        aligned = torch.zeros_like(tok32)
        for i, lag in enumerate(self.lags):
            shifted = torch.cat([tok32[:, :1].expand(-1, lag + 1, -1, -1), tok32], dim=1)[:, :target_steps] if lag > 0 else tok32
            aligned = aligned + weights[..., i : i + 1] * shifted
        flow.response_lag_weights = weights
        flow.lag_aligned_tokens = aligned
        return flow

