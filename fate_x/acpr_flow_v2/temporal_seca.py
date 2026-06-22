from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import SemanticReasonMemory


@dataclass
class SECADiagnostics:
    attention: Tensor
    gate: Tensor
    token_delta: Optional[Tensor] = None
    image_hidden_max_diff: Optional[Tensor] = None
    generation_segment: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        aliases = {
            "token_reason_attention": "attention",
            "token_delta": "token_delta",
            "image_hidden_max_diff": "image_hidden_max_diff",
            "gate": "gate",
            "generation_segment": "generation_segment",
        }
        attr = aliases.get(key, key)
        return getattr(self, attr, default)


class TemporalSECAV2(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.query_action = nn.Linear(hidden_dim, hidden_dim)
        self.query_explanation = nn.Linear(hidden_dim, hidden_dim)
        self.out_action = nn.Linear(hidden_dim, hidden_dim)
        self.out_explanation = nn.Linear(hidden_dim, hidden_dim)
        self.gate_action = nn.Parameter(torch.tensor(0.0))
        self.gate_explanation = nn.Parameter(torch.tensor(0.0))

    def _segment_masks(
        self,
        hidden: Tensor,
        token_type_ids: Optional[Tensor],
        text_len: Optional[int],
        generation_segment: Optional[str],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        bsz, seq_len = hidden.shape[:2]
        action_mask = hidden.new_zeros((bsz, seq_len, 1))
        explanation_mask = hidden.new_zeros((bsz, seq_len, 1))
        text_mask = hidden.new_zeros((bsz, seq_len, 1))

        if text_len is None:
            limit = seq_len if token_type_ids is None else min(seq_len, token_type_ids.shape[1])
        else:
            limit = min(seq_len, max(0, int(text_len)))
        if limit <= 0:
            return action_mask, explanation_mask, text_mask
        text_mask[:, :limit] = 1.0

        segment = (generation_segment or "").lower()
        if segment in {"description", "action"}:
            action_mask[:, :limit] = 1.0
            return action_mask, explanation_mask, text_mask
        if segment == "explanation":
            explanation_mask[:, :limit] = 1.0
            return action_mask, explanation_mask, text_mask

        if token_type_ids is not None:
            type_len = min(limit, token_type_ids.shape[1])
            type_ids = token_type_ids[:, :type_len].to(hidden.device)
            action_mask[:, :type_len, 0] = (type_ids == 0).to(hidden.dtype)
            explanation_mask[:, :type_len, 0] = (type_ids != 0).to(hidden.dtype)
        else:
            # Safe fallback: without segment/type evidence, treat the hook as
            # explanation-only. This avoids silently rewriting description
            # tokens during semantic recovery.
            explanation_mask[:, :limit] = 1.0
        return action_mask, explanation_mask, text_mask

    def forward(self, hidden: Tensor, memory: SemanticReasonMemory, token_type_ids: Optional[Tensor] = None, text_len: Optional[int] = None, generation_segment: Optional[str] = None) -> Tuple[Tensor, SECADiagnostics]:
        action_mask, explanation_mask, text_mask = self._segment_masks(hidden, token_type_ids, text_len, generation_segment)

        q_action = self.query_action(hidden)
        q_explanation = self.query_explanation(hidden)
        scores_action = torch.einsum("bld,bmd->blm", q_action, memory.values) / math.sqrt(q_action.shape[-1])
        scores_explanation = torch.einsum("bld,bmd->blm", q_explanation, memory.values) / math.sqrt(q_explanation.shape[-1])
        memory_mask = ~memory.mask.unsqueeze(1)
        scores_action = scores_action.masked_fill(memory_mask, -1e4)
        scores_explanation = scores_explanation.masked_fill(memory_mask, -1e4)
        attn_action = torch.softmax(scores_action, dim=-1)
        attn_explanation = torch.softmax(scores_explanation, dim=-1)
        ctx_action = torch.einsum("blm,bmd->bld", attn_action, memory.values)
        ctx_explanation = torch.einsum("blm,bmd->bld", attn_explanation, memory.values)
        gate_action = torch.tanh(self.gate_action) * 0.08
        gate_explanation = torch.tanh(self.gate_explanation) * 0.25
        delta_action = gate_action * self.out_action(ctx_action) * action_mask
        delta_explanation = gate_explanation * self.out_explanation(ctx_explanation) * explanation_mask
        delta = delta_action + delta_explanation
        image_hidden_max_diff = hidden.new_tensor(0.0)
        non_text = text_mask <= 0
        if non_text.any():
            image_hidden_max_diff = delta.masked_select(non_text.expand_as(delta)).abs().max().detach()
        enhanced = hidden + delta
        return enhanced, SECADiagnostics(
            attention=attn_action * action_mask + attn_explanation * explanation_mask,
            gate=torch.stack([gate_action.detach(), gate_explanation.detach()]),
            token_delta=delta.detach(),
            image_hidden_max_diff=image_hidden_max_diff,
            generation_segment=generation_segment,
        )


TemporalSECA = TemporalSECAV2
