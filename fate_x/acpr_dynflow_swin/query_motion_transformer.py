from __future__ import annotations

import torch
from torch import Tensor, nn

from .types import MotionTransformerOutput


class QueryMotionTransformer(nn.Module):
    """BERT-base-capacity query transformer for 32-step control prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        nhead: int = 12,
        intermediate_size: int = 3072,
        output_queries: int = 32,
    ):
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by nhead={nhead}")
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=intermediate_size,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.query = nn.Parameter(torch.randn(output_queries, hidden_dim) * 0.02)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, temporal_global: Tensor, semantic_tokens: Tensor) -> MotionTransformerOutput:
        bsz = temporal_global.shape[0]
        semantic_flat = semantic_tokens.reshape(bsz, -1, semantic_tokens.shape[-1])
        context = torch.cat([temporal_global, semantic_flat], dim=1)
        context_hidden = self.input_proj(context)
        queries = self.query.unsqueeze(0).expand(bsz, -1, -1)
        encoded = self.encoder(torch.cat([queries, context_hidden], dim=1))
        query_hidden = encoded[:, : self.query.shape[0]]
        pred = self.out(query_hidden)
        return MotionTransformerOutput(
            query_hidden=query_hidden,
            global_prediction_normalized=pred,
            source_attention=None,
        )
