from __future__ import annotations

import torch
from torch import Tensor, nn

from .activations import entmax15
from .temporal_seca import scaled_gradient


class ReasonControlAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 768, signals: int = 2, max_residual_std_fraction: float = 0.15,
                 evidence_grad_scale: float = 0.25) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.delta = nn.Linear(hidden_dim, signals)
        self.gate_raw = nn.Parameter(torch.zeros(signals))
        self.max_residual_std_fraction = max_residual_std_fraction
        self.evidence_grad_scale = evidence_grad_scale

    def forward(self, base_prediction: Tensor, control_hidden: Tensor, reason_memory: Tensor,
                signal_scale: Tensor | float = 1.0) -> dict[str, Tensor]:
        mem = scaled_gradient(reason_memory, self.evidence_grad_scale)
        q = self.query(control_hidden)
        k = self.key(mem)
        v = self.value(mem)
        attn = entmax15(torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5), dim=-1)
        context = torch.matmul(attn, v)
        scale = torch.as_tensor(signal_scale, device=base_prediction.device, dtype=base_prediction.dtype)
        delta = torch.tanh(self.delta(context)) * self.max_residual_std_fraction * scale
        gate = torch.tanh(self.gate_raw).to(base_prediction.dtype)
        final = base_prediction + delta * gate.view(1, 1, -1)
        return {"control_final_prediction": final, "control_delta": delta * gate.view(1, 1, -1), "control_reason_attention": attn}
