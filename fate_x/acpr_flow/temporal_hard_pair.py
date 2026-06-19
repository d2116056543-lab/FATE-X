from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TemporalHardPairQueue(nn.Module):
    def __init__(self, hidden_dim: int = 768, queue_size: int = 4096, margin: float = 0.20,
                 max_pairs_per_batch: int = 64, pair_budget_ratio: float = 0.08) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.queue_size = int(queue_size)
        self.margin = float(margin)
        self.max_pairs_per_batch = int(max_pairs_per_batch)
        self.pair_budget_ratio = float(pair_budget_ratio)
        self.register_buffer("reason_target_queue", torch.empty(0, hidden_dim), persistent=False)
        self.register_buffer("action_queue", torch.empty(0, hidden_dim), persistent=False)

    @torch.no_grad()
    def enqueue(self, reason_target: Tensor, action_embedding: Tensor) -> None:
        self.reason_target_queue = torch.cat([self.reason_target_queue, reason_target.detach().cpu()], dim=0)[-self.queue_size:]
        self.action_queue = torch.cat([self.action_queue, action_embedding.detach().cpu()], dim=0)[-self.queue_size:]

    def forward(self, predicted_reason: Tensor, reason_target: Tensor, action_embedding: Tensor,
                base_loss: Tensor | None = None) -> dict[str, Tensor]:
        pred = F.normalize(self.proj(predicted_reason), dim=-1)
        target = F.normalize(reason_target, dim=-1)
        if self.reason_target_queue.numel() == 0:
            raw = pred.new_zeros(())
            active = pred.new_zeros(())
        else:
            q_reason = self.reason_target_queue.to(pred.device, pred.dtype)
            q_action = self.action_queue.to(pred.device, pred.dtype)
            sim_action = torch.matmul(F.normalize(action_embedding, dim=-1), F.normalize(q_action, dim=-1).T)
            sim_reason = torch.matmul(target, F.normalize(q_reason, dim=-1).T)
            eligible = (sim_action >= 0.75) & (sim_reason <= 0.35)
            if eligible.any():
                neg_idx = eligible.float().argmax(dim=1)
                neg = F.normalize(q_reason[neg_idx], dim=-1)
                raw_each = (self.margin - (pred * target).sum(-1) + (pred * neg).sum(-1)).clamp_min(0)
                raw = raw_each.mean()
                active = eligible.any(dim=1).float().mean()
            else:
                raw = pred.new_zeros(())
                active = pred.new_zeros(())
        if base_loss is None:
            budgeted = raw
        else:
            budgeted = torch.minimum(raw, base_loss.detach() * self.pair_budget_ratio)
        return {
            "hardpair_raw_loss": raw,
            "hardpair_budgeted_loss": budgeted,
            "active_pair_rate": active,
            "candidate_count": pred.new_tensor(float(self.reason_target_queue.shape[0])),
        }
