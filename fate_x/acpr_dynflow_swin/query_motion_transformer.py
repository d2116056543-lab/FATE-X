from __future__ import annotations

import torch
from torch import Tensor, nn

from .types import MotionTransformerOutput


class QueryMotionTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=max(1, min(8, hidden_dim // 8)), batch_first=True)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.query = nn.Parameter(torch.randn(32, hidden_dim) * 0.02)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, temporal_global: Tensor, semantic_tokens: Tensor) -> MotionTransformerOutput:
        bsz = temporal_global.shape[0]
        context = torch.cat([temporal_global, semantic_tokens.reshape(bsz, -1, semantic_tokens.shape[-1])], dim=1)
        hidden = self.encoder(self.input_proj(context))
        pooled = hidden.mean(dim=1, keepdim=True)
        query_hidden = pooled + self.query.unsqueeze(0)
        pred = self.out(query_hidden)
        return MotionTransformerOutput(query_hidden=query_hidden, global_prediction_normalized=pred, source_attention=None)
