from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FlowTraceLoss(nn.Module):
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        self.weights = weights or {}

    def forward(self, bundle, text_loss: Tensor | None = None, control_loss: Tensor | None = None,
                anchor_target: Tensor | None = None, reason_target: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        device = bundle.state_memory.device
        total = torch.zeros((), device=device)
        logs: dict[str, Tensor] = {}
        if text_loss is not None:
            total = total + text_loss
            logs["text"] = text_loss.detach()
        if control_loss is not None:
            total = total + control_loss
            logs["control"] = control_loss.detach()
        if anchor_target is not None:
            q = bundle.reason_state_distribution.clamp_min(1e-6)
            p = anchor_target.to(q.device).clamp_min(1e-6)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            loss = F.kl_div(q.log(), p, reduction="batchmean")
            total = total + self.weights.get("anchor", 0.05) * loss
            logs["anchor"] = loss.detach()
        if reason_target is not None:
            loss = 1.0 - F.cosine_similarity(bundle.reason_state, reason_target.to(device), dim=-1).mean()
            total = total + self.weights.get("reason", 0.05) * loss
            logs["reason_state"] = loss.detach()
        diversity = (bundle.state_memory[:, :, None] - bundle.state_memory[:, None]).pow(2).mean()
        state_div = torch.exp(-diversity)
        total = total + self.weights.get("state_diversity", 0.002) * state_div
        logs["state_diversity"] = state_div.detach()
        return total, logs
