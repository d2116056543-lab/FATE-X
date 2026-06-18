from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenPMTAdapter(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int = 256, rank: int = 32,
                 action_gate_max: float = 0.15, explanation_gate_max: float = 0.30) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.rank = rank
        self.action_gate_max = action_gate_max
        self.explanation_gate_max = explanation_gate_max
        self.token_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.state_proj = nn.Linear(state_dim, rank, bias=False)
        self.reason_proj = nn.Linear(state_dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.state_to_hidden = nn.Linear(state_dim, hidden_dim, bias=False)
        self.exec_count = 0
        self.last_routing: Tensor | None = None
        self.last_delta: Tensor | None = None
        self.last_gate: Tensor | None = None

    def forward(self, hidden_states: Tensor, state_memory: Tensor, reason_state: Tensor,
                token_type_ids: Tensor | None = None, scale: float | Tensor = 1.0) -> tuple[Tensor, dict[str, Tensor]]:
        if state_memory is None or reason_state is None:
            return hidden_states, {}
        self.exec_count += 1
        attn_logits = torch.matmul(hidden_states, self.state_to_hidden(state_memory).transpose(-1, -2))
        routing = torch.softmax(attn_logits, dim=-1)
        context = torch.matmul(routing, state_memory)
        u = self.token_proj(hidden_states) * self.state_proj(context) * self.reason_proj(reason_state).unsqueeze(1)
        delta = torch.tanh(self.out_proj(u))
        if token_type_ids is None:
            gate = torch.full(hidden_states.shape[:2], self.action_gate_max, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            gate = torch.where(token_type_ids.to(hidden_states.device) == 1,
                               torch.tensor(self.explanation_gate_max, device=hidden_states.device, dtype=hidden_states.dtype),
                               torch.tensor(self.action_gate_max, device=hidden_states.device, dtype=hidden_states.dtype))
        if not torch.is_tensor(scale):
            scale = torch.tensor(float(scale), device=hidden_states.device, dtype=hidden_states.dtype)
        out = hidden_states + scale * gate.unsqueeze(-1) * delta
        self.last_routing = routing
        self.last_delta = delta
        self.last_gate = gate
        return out, {"token_state_routing": routing, "pmt_delta": delta, "pmt_gate": gate}
