from __future__ import annotations

from torch import Tensor, nn


class ContributionReasonAdapter(nn.Module):
    def __init__(self, factor_dim: int, text_dim: int):
        super().__init__()
        self.proj = nn.Linear(factor_dim, text_dim)

    def forward(self, text_hidden: Tensor, factor_summary: Tensor) -> Tensor:
        return text_hidden + self.proj(factor_summary).unsqueeze(1)
