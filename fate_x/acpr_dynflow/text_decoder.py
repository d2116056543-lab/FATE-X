from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import DecisionLedger, DynFlowTextOutput, TrafficFlowState


class DynFlowTextDecoder(nn.Module):
    def __init__(self, text_dim: int = 768, factor_dim: int = 256, vocab_size: int = 30522):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.factor_to_text = nn.Linear(factor_dim + 2, text_dim)
        self.action_lm = nn.Linear(text_dim, vocab_size)
        self.explanation_lm = nn.Linear(text_dim, vocab_size)
        self.vocab_size = vocab_size

    def _loss(self, logits: Tensor, target: Tensor | None) -> Tensor:
        if target is None or target.numel() == 0:
            return logits.sum() * 0.0
        flat_target = target.reshape(-1)
        valid = flat_target.ge(0)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        flat_logits = logits.reshape(-1, logits.shape[-1])[: flat_target.numel()]
        return F.cross_entropy(flat_logits[valid], flat_target[valid].clamp_max(logits.shape[-1] - 1))

    def forward(self, input_ids: Tensor | None, masked_ids: Tensor | None, flow: TrafficFlowState, ledger: DecisionLedger) -> DynFlowTextOutput:
        b = flow.factor_tokens.shape[0]
        length = input_ids.shape[1] if input_ids is not None else 30
        ids = input_ids if input_ids is not None else torch.zeros(b, length, dtype=torch.long, device=flow.factor_tokens.device)
        base = self.token_embed(ids.clamp_min(0).clamp_max(self.vocab_size - 1))
        factor_context = torch.cat([flow.lag_aligned_tokens.mean(2), ledger.final_prediction_normalized.detach()], dim=-1)
        if factor_context.shape[1] != length:
            factor_context = torch.nn.functional.interpolate(factor_context.permute(0, 2, 1), size=length, mode="linear", align_corners=False).permute(0, 2, 1)
        ctx = base + self.factor_to_text(factor_context)
        action_logits = self.action_lm(ctx)
        explanation_logits = self.explanation_lm(ctx)
        midpoint = max(1, length // 2)
        action_loss = self._loss(action_logits[:, :midpoint], masked_ids)
        explanation_loss = self._loss(explanation_logits[:, midpoint:], masked_ids)
        attn_base = flow.factor_probs
        if attn_base.shape[1] != length:
            attn_base = torch.nn.functional.interpolate(attn_base.permute(0, 2, 1), size=length, mode="linear", align_corners=False).permute(0, 2, 1)
        return DynFlowTextOutput(
            action_logits=action_logits,
            explanation_logits=explanation_logits,
            action_loss=action_loss,
            explanation_loss=explanation_loss,
            explanation_to_factor_attention=torch.softmax(attn_base, dim=-1),
            action_to_factor_attention=torch.softmax(attn_base, dim=-1),
            generated_action=None,
            generated_explanation=None,
        )

