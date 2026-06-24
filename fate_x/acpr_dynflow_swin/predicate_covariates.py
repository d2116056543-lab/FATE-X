from __future__ import annotations

from torch import Tensor, nn


class PredicateCovariateAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, covariates: Tensor) -> Tensor:
        return self.proj(covariates)
