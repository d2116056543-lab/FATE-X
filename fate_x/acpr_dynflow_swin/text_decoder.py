from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import DynFlowSwinTextOutput


def _masked_ce(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    valid = mask.bool()
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid].long())


class DynFlowSwinTextDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 768, vocab_size: int = 30522, factor_dim: int = 256):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.factor_proj = nn.Linear(factor_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.factor_attention = nn.Linear(hidden_dim, 13)

    def forward(
        self,
        input_ids: Tensor,
        masked_pos: Tensor,
        masked_ids: Tensor,
        text_hidden: Tensor,
        factor_tokens: Tensor,
    ) -> DynFlowSwinTextOutput:
        hidden = self.token_embed(input_ids.clamp_min(0)) + text_hidden
        factor_context = self.factor_proj(factor_tokens.mean(dim=1))
        hidden = hidden + factor_context.mean(dim=1, keepdim=True)
        logits = self.lm_head(hidden)
        action_mask = masked_pos.bool() & (torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1) < 15)
        explanation_mask = masked_pos.bool() & ~action_mask
        action_loss = _masked_ce(logits, masked_ids, action_mask)
        explanation_loss = _masked_ce(logits, masked_ids, explanation_mask)
        attn = torch.softmax(self.factor_attention(hidden), dim=-1)
        return DynFlowSwinTextOutput(
            total_mlm_loss=action_loss + explanation_loss,
            action_loss=action_loss,
            explanation_loss=explanation_loss,
            action_logits=logits[:, :15],
            explanation_logits=logits[:, 15:30],
            action_to_factor_attention=attn[:, :15].mean(dim=1),
            explanation_to_factor_attention=attn[:, 15:30].mean(dim=1),
            generated_action=None,
            generated_explanation=None,
        )
