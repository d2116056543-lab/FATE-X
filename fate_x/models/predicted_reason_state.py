from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PredictedReasonState(nn.Module):
    def __init__(self, state_dim: int = 256) -> None:
        super().__init__()
        self.query = nn.Linear(state_dim, 1)
        self.proj = nn.Linear(state_dim, state_dim)

    def forward(self, state_memory: Tensor) -> dict[str, Tensor]:
        weights = torch.softmax(self.query(state_memory).squeeze(-1), dim=-1)
        reason = torch.einsum("bk,bkd->bd", weights, state_memory)
        reason = F.normalize(self.proj(reason), dim=-1)
        return {"reason_state": reason, "reason_state_distribution": weights}
