from __future__ import annotations

from torch import Tensor, nn


class PrefixFutureHead(nn.Module):
    def __init__(self, state_dim: int = 256, signals: int = 2, future_steps: int = 8) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.signals = signals
        self.head = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, future_steps * signals))

    def forward(self, global_state: Tensor) -> Tensor:
        return self.head(global_state).reshape(global_state.shape[0], self.future_steps, self.signals)
