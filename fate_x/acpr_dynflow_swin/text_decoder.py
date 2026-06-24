from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional

from src.layers.bert import BertForImageCaptioning

from .types import DynFlowSwinTextOutput


def _masked_ce(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    valid = mask.bool()
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid].long())


def _gather_masked_logits(sequence_logits: Tensor, masked_pos: Tensor) -> Tensor:
    max_index = sequence_logits.shape[1] - 1
    positions = masked_pos.long().clamp_min(0).clamp_max(max_index)
    gather_index = positions.unsqueeze(-1).expand(-1, -1, sequence_logits.shape[-1])
    return sequence_logits.gather(dim=1, index=gather_index)


def _binary_masked_ce(sequence_logits: Tensor, labels: Tensor, mask: Tensor, offsets: Optional[Tensor] = None) -> Tensor:
    losses = []
    for batch_index in range(sequence_logits.shape[0]):
        valid_positions = mask[batch_index].bool()
        count = int(valid_positions.sum().detach().cpu())
        if count <= 0:
            continue
        offset = int(offsets[batch_index].detach().cpu()) if offsets is not None else 0
        target = labels[batch_index, offset : offset + count].long()
        losses.append(F.cross_entropy(sequence_logits[batch_index, valid_positions], target))
    if not losses:
        return sequence_logits.sum() * 0.0
    return torch.stack(losses).mean()


class DynFlowSwinTextDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 30522,
        factor_dim: int = 256,
        bert_captioner: BertForImageCaptioning | None = None,
    ):
        super().__init__()
        self.bert_captioner = bert_captioner
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.factor_proj = nn.Linear(factor_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.factor_attention = nn.Linear(hidden_dim, 13)

    def generate(
        self,
        img_feats: Tensor,
        attention_mask: Tensor,
        masked_pos: Tensor,
        token_type_ids: Tensor,
        use_sep_cap: bool = True,
        max_length: int = 30,
    ) -> Tensor:
        if self.bert_captioner is None:
            raise RuntimeError("ADAPT BertForImageCaptioning generation requires bert_captioner")
        return self.bert_captioner.generate(
            img_feats=img_feats,
            attention_mask=attention_mask,
            masked_pos=masked_pos,
            token_type_ids=token_type_ids,
            max_length=max_length,
            use_sep_cap=use_sep_cap,
        )

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
        if masked_pos.shape[1] == input_ids.shape[1] and int(masked_pos.max().detach().cpu()) <= 1:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
            valid_mask = masked_pos.bool()
            action_mask = valid_mask & positions.lt(15)
            explanation_mask = valid_mask & positions.ge(15)
            action_offsets = action_mask.sum(dim=1)
            action_loss = _binary_masked_ce(logits, masked_ids, action_mask)
            explanation_loss = _binary_masked_ce(logits, masked_ids, explanation_mask, offsets=action_offsets)
        else:
            masked_logits = _gather_masked_logits(logits, masked_pos)
            valid_mask = masked_ids.ge(0)
            action_mask = valid_mask & masked_pos.lt(15)
            explanation_mask = valid_mask & masked_pos.ge(15)
            action_loss = _masked_ce(masked_logits, masked_ids, action_mask)
            explanation_loss = _masked_ce(masked_logits, masked_ids, explanation_mask)
        attn = torch.softmax(self.factor_attention(hidden), dim=-1)
        return DynFlowSwinTextOutput(
            total_mlm_loss=action_loss + explanation_loss,
            action_loss=action_loss,
            explanation_loss=explanation_loss,
            action_logits=logits[:, :15],
            explanation_logits=logits[:, 15:30],
            action_to_factor_attention=attn[:, :15].mean(dim=1),
            explanation_to_factor_attention=attn[:, 15:30].mean(dim=1),
            generated_action=[],
            generated_explanation=[],
        )
