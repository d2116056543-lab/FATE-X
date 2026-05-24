from __future__ import annotations

import torch
from torch import nn


DEFAULT_EVENTS = [
    "longitudinal_motion",
    "lateral_motion",
    "traffic_control",
    "obstacle",
    "lane_road",
    "rule_constraint",
    "uncertainty",
    "background_context",
]


class TemporalEvidenceMemory(nn.Module):
    """Event-query memory over reduced video tokens."""

    def __init__(self, dim: int, num_heads: int = 8, event_names: list[str] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.event_names = event_names or DEFAULT_EVENTS
        self.event_queries = nn.Parameter(torch.randn(len(self.event_names), dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, video_tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must be [B,N,D]")
        b = video_tokens.shape[0]
        q = self.event_queries.unsqueeze(0).expand(b, -1, -1)
        tokens, attn = self.attn(q, video_tokens, video_tokens, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        return {"event_tokens": self.norm(tokens), "event_attention": attn}