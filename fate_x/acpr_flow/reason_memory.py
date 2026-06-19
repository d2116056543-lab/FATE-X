from __future__ import annotations

import torch
from torch import Tensor, nn


class ReasonMemory(nn.Module):
    def __init__(self, state_dim: int = 256, hidden_dim: int = 768, num_predicates: int = 32, num_flow: int = 13) -> None:
        super().__init__()
        self.num_predicates = num_predicates
        self.num_flow = num_flow
        self.local_proj = nn.Linear(state_dim, hidden_dim)
        self.flow_proj = nn.Linear(state_dim * 2, hidden_dim)
        self.local_semantic = nn.Parameter(torch.randn(num_predicates, hidden_dim) * 0.02)
        self.flow_semantic = nn.Parameter(torch.randn(num_flow, hidden_dim) * 0.02)
        self.null_reason = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, predicate_descriptor: Tensor, flow_tokens: Tensor, flow_to_predicate_attention: Tensor, acpr_x: bool = False) -> dict[str, Tensor]:
        b = predicate_descriptor.shape[0]
        local = self.norm(self.local_proj(predicate_descriptor) + self.local_semantic.unsqueeze(0))
        support = torch.einsum("bkp,bpd->bkd", flow_to_predicate_attention, predicate_descriptor)
        flow = self.norm(self.flow_proj(torch.cat([flow_tokens, support], dim=-1)) + self.flow_semantic.unsqueeze(0))
        if acpr_x:
            flow = flow.detach() * 0.0
            flow_mask = torch.zeros(b, self.num_flow, dtype=torch.bool, device=predicate_descriptor.device)
        else:
            flow_mask = torch.ones(b, self.num_flow, dtype=torch.bool, device=predicate_descriptor.device)
        null = self.null_reason.expand(b, -1, -1).to(dtype=local.dtype)
        memory = torch.cat([local, flow, null], dim=1)
        mask = torch.cat([
            torch.ones(b, self.num_predicates, dtype=torch.bool, device=predicate_descriptor.device),
            flow_mask,
            torch.ones(b, 1, dtype=torch.bool, device=predicate_descriptor.device),
        ], dim=1)
        global_state = (memory * mask.to(memory.dtype).unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return {
            "local_reason_memory": local,
            "flow_reason_memory": flow,
            "null_reason_memory": null,
            "reason_memory": memory,
            "reason_memory_mask": mask,
            "global_reason_state": global_state,
            "reason_memory_types": ["local"] * self.num_predicates + ["flow"] * self.num_flow + ["null"],
        }
