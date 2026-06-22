from __future__ import annotations

from pathlib import Path
import os

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import DecisionLedger, DynFlowTextOutput, TrafficFlowState


class DynFlowTextDecoder(nn.Module):
    """BERT-base text path with separate action/explanation heads.

    When a local BERT directory is supplied, the module loads `BertModel`
    directly and freezes the bottom 8 encoder layers, keeping the top 4 layers
    trainable as required by the DynFlow plan. A lightweight fallback is only
    used for unit tests without assets.
    """

    def __init__(self, text_dim: int = 768, factor_dim: int = 256, vocab_size: int = 30522, bert_dir: str | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.bert_dir = bert_dir or ""
        self.bert_loaded = False
        self.bert_load_error = ""

        if bert_dir and Path(bert_dir).exists():
            try:
                os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
                from transformers import BertModel

                self.bert = BertModel.from_pretrained(bert_dir, local_files_only=True)
                text_dim = int(self.bert.config.hidden_size)
                self.vocab_size = int(self.bert.config.vocab_size)
                self.bert_loaded = True
                for idx, layer in enumerate(self.bert.encoder.layer):
                    requires_grad = idx >= max(0, len(self.bert.encoder.layer) - 4)
                    for p in layer.parameters():
                        p.requires_grad = requires_grad
                for p in self.bert.embeddings.parameters():
                    p.requires_grad = False
                for p in self.bert.pooler.parameters():
                    p.requires_grad = False
            except Exception as exc:  # pragma: no cover - exercised by preflight on asset failures
                self.bert_load_error = f"{type(exc).__name__}: {exc}"
                self.bert = None
        else:
            self.bert = None

        if self.bert is None:
            self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.factor_to_text = nn.Linear(factor_dim + 2, text_dim)
        self.action_lm = nn.Linear(text_dim, self.vocab_size)
        self.explanation_lm = nn.Linear(text_dim, self.vocab_size)

    def _loss(self, logits: Tensor, target: Tensor | None) -> Tensor:
        if target is None or target.numel() == 0:
            return logits.sum() * 0.0
        flat_target = target.reshape(-1)
        valid = flat_target.ge(0)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        flat_logits = logits.reshape(-1, logits.shape[-1])[: flat_target.numel()]
        return F.cross_entropy(flat_logits[valid], flat_target[valid].clamp_max(logits.shape[-1] - 1))

    def _masked_position_loss(
        self,
        logits: Tensor,
        masked_pos: Tensor | None,
        masked_ids: Tensor | None,
        start: int,
        end: int,
    ) -> Tensor:
        if masked_ids is None or masked_ids.numel() == 0:
            return logits.sum() * 0.0
        if masked_pos is None or masked_pos.numel() == 0:
            target = masked_ids[:, start:end] if masked_ids.dim() == 2 and masked_ids.shape[1] >= end else masked_ids
            return self._loss(logits[:, start:end], target)
        positions = masked_pos.to(logits.device).long()
        targets = masked_ids.to(logits.device).long()
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(logits.shape[0], -1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0).expand(logits.shape[0], -1)
        valid = targets.ge(0) & positions.ge(start) & positions.lt(end) & positions.lt(logits.shape[1])
        if not bool(valid.any()):
            return logits.sum() * 0.0
        gather_pos = positions.clamp(0, logits.shape[1] - 1).unsqueeze(-1).expand(-1, -1, logits.shape[-1])
        gathered_logits = logits.gather(1, gather_pos)
        return F.cross_entropy(gathered_logits[valid], targets[valid].clamp_max(logits.shape[-1] - 1))

    def _base_hidden(self, ids: Tensor) -> Tensor:
        if self.bert is not None:
            attention = torch.ones_like(ids, dtype=torch.long)
            return self.bert(input_ids=ids.clamp_min(0).clamp_max(self.vocab_size - 1), attention_mask=attention).last_hidden_state
        return self.token_embed(ids.clamp_min(0).clamp_max(self.vocab_size - 1))

    def forward(self, input_ids: Tensor | None, masked_pos: Tensor | None, masked_ids: Tensor | None, flow: TrafficFlowState, ledger: DecisionLedger, visual_tokens: Tensor | None = None) -> DynFlowTextOutput:
        b = flow.factor_tokens.shape[0]
        length = input_ids.shape[1] if input_ids is not None else 30
        ids = input_ids if input_ids is not None else torch.zeros(b, length, dtype=torch.long, device=flow.factor_tokens.device)
        base = self._base_hidden(ids)
        factor_context = torch.cat([flow.lag_aligned_tokens.mean(2), ledger.final_prediction_normalized.detach()], dim=-1)
        if factor_context.shape[1] != length:
            factor_context = F.interpolate(factor_context.permute(0, 2, 1), size=length, mode="linear", align_corners=False).permute(0, 2, 1)
        ctx = base + self.factor_to_text(factor_context)
        if visual_tokens is not None:
            if visual_tokens.shape[1] != length:
                visual_tokens = F.interpolate(visual_tokens.permute(0, 2, 1), size=length, mode="linear", align_corners=False).permute(0, 2, 1)
            ctx = ctx + visual_tokens
        action_logits = self.action_lm(ctx)
        explanation_logits = self.explanation_lm(ctx)
        midpoint = max(1, length // 2)
        action_loss = self._masked_position_loss(action_logits, masked_pos, masked_ids, 0, midpoint)
        explanation_loss = self._masked_position_loss(explanation_logits, masked_pos, masked_ids, midpoint, length)
        attn_base = flow.factor_probs
        if attn_base.shape[1] != length:
            attn_base = F.interpolate(attn_base.permute(0, 2, 1), size=length, mode="linear", align_corners=False).permute(0, 2, 1)
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
