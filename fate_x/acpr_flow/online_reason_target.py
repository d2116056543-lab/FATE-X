from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def masked_mean_embeddings(embeddings: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
    return (embeddings * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def build_action_residual_reason_target(
    word_embeddings: Tensor,
    action_token_mask: Tensor,
    reason_token_mask: Tensor,
    residual_strength: float = 1.0,
) -> Tensor:
    action = F.normalize(masked_mean_embeddings(word_embeddings.detach(), action_token_mask), dim=-1)
    reason = F.normalize(masked_mean_embeddings(word_embeddings.detach(), reason_token_mask), dim=-1)
    projection = (reason * action).sum(dim=-1, keepdim=True) * action
    residual = reason - residual_strength * projection
    return F.normalize(residual, dim=-1)
