from __future__ import annotations

from torch import Tensor, nn


class GlobalDecisionStream(nn.Module):
    def __init__(self, dim: int = 256, decision_dim: int = 512):
        super().__init__()
        self.gru = nn.GRU(dim, decision_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(decision_dim, 2)

    def forward(self, global_sequence: Tensor) -> tuple[Tensor, Tensor]:
        state, _ = self.gru(global_sequence)
        return self.out(state), state

