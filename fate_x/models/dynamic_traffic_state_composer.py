from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicTrafficStateComposer(nn.Module):
    def __init__(self, state_dim: int = 256, num_states: int = 8, num_tracks: int = 12,
                 temporal_scales: tuple[int, ...] = (1, 2, 4), heads: int = 4) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_tracks = num_tracks
        self.temporal_scales = temporal_scales
        descriptor_dim = state_dim * (2 + len(temporal_scales)) + 3
        self.desc_proj = nn.Linear(descriptor_dim, state_dim)
        self.state_queries = nn.Parameter(torch.randn(num_states, state_dim) * 0.02)
        self.track_attn = nn.MultiheadAttention(state_dim, heads, batch_first=True)
        enc = nn.TransformerEncoderLayer(state_dim, heads, dim_feedforward=state_dim * 2, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc, num_layers=1)
        enc2 = nn.TransformerEncoderLayer(state_dim, heads, dim_feedforward=state_dim * 2, batch_first=True)
        self.state_encoder = nn.TransformerEncoder(enc2, num_layers=1)
        self.score = nn.Linear(state_dim, 1)

    def _delta(self, x: Tensor, scale: int) -> Tensor:
        out = torch.zeros_like(x)
        out[:, scale:] = x[:, scale:] - x[:, :-scale]
        return out

    def forward(self, track_tokens: Tensor, track_attention: Tensor, relative_motion: Tensor,
                track_unmatched_mass: Tensor) -> dict[str, Tensor]:
        b, t, l, d = track_tokens.shape
        rel = torch.zeros(b, t, l, 2, device=track_tokens.device, dtype=track_tokens.dtype)
        if relative_motion.numel() > 0:
            rel[:, 1:] = relative_motion
        pieces = [track_tokens, rel.new_zeros(b, t, l, d)]
        pieces[1][..., :2] = rel
        for s in self.temporal_scales:
            pieces.append(self._delta(track_tokens, s))
        occ = track_unmatched_mass.unsqueeze(-1)
        desc = torch.cat(pieces + [occ, rel], dim=-1)
        desc = self.desc_proj(desc)
        q = self.state_queries.unsqueeze(0).expand(b * t, -1, -1)
        kv = desc.reshape(b * t, l, d)
        z, weights = self.track_attn(q, kv, kv, need_weights=True)
        z = z.reshape(b, t, self.num_states, d)
        weights = weights.reshape(b, t, self.num_states, l)
        temporal = self.temporal_encoder(z.permute(0, 2, 1, 3).reshape(b * self.num_states, t, d))
        temporal = temporal.reshape(b, self.num_states, t, d).permute(0, 2, 1, 3)
        inter = self.state_encoder(temporal.reshape(b * t, self.num_states, d))
        state_tokens_temporal = inter.reshape(b, t, self.num_states, d)
        state_memory = state_tokens_temporal.mean(dim=1)
        state_scores = self.score(state_tokens_temporal).squeeze(-1)
        maps = torch.einsum("btkl,btlhw->btkhw", weights, track_attention)
        return {
            "state_tokens_temporal": state_tokens_temporal,
            "state_memory": state_memory,
            "state_scores": state_scores,
            "state_track_weights": weights,
            "state_evidence_maps": maps,
        }
