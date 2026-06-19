from __future__ import annotations

import torch
from torch import Tensor, nn

from .activations import entmax15
from .region_priors import ACPR_PREDICATE_NAMES, build_region_prior_grid


class TemporalPredicateEmbeddingField(nn.Module):
    def __init__(self, state_dim: int = 256, num_predicates: int = 32) -> None:
        super().__init__()
        self.num_predicates = num_predicates
        self.queries = nn.Parameter(torch.randn(num_predicates, state_dim) * 0.02)
        self.token_proj = nn.Linear(state_dim * 6 + 3, state_dim)
        self.presence_head = nn.Linear(state_dim, 1)

    def _descriptor(self, tokens: Tensor, rel_motion: Tensor, confidence: Tensor) -> dict[str, Tensor]:
        now = tokens[:, -1]
        history = tokens.mean(dim=1)
        trend = torch.zeros_like(now)
        if tokens.shape[1] > 1:
            weights = torch.linspace(0.25, 1.0, tokens.shape[1] - 1, device=tokens.device, dtype=tokens.dtype)
            delta = tokens[:, 1:] - tokens[:, :-1]
            trend = (delta * weights.view(1, -1, 1, 1)).mean(dim=1)
        volatility = torch.zeros_like(now)
        if tokens.shape[1] > 2:
            volatility = (tokens[:, 2:] - 2 * tokens[:, 1:-1] + tokens[:, :-2]).abs().mean(dim=1)
        motion = torch.zeros(tokens.shape[0], self.num_predicates, 2, device=tokens.device, dtype=tokens.dtype)
        if rel_motion.numel() > 0:
            motion = rel_motion.mean(dim=1)
        conf = confidence.mean(dim=1).unsqueeze(-1)
        pieces = [now, history, trend, volatility, tokens.std(dim=1), tokens.amax(dim=1)]
        desc = self.token_proj(torch.cat(pieces + [motion, conf], dim=-1))
        return {
            "now": now,
            "history": history,
            "trend": trend,
            "volatility": volatility,
            "motion": motion,
            "confidence": conf,
            "descriptor": desc,
        }

    def forward(self, fused_grid: Tensor, transport: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        b, t, h, w, d = fused_grid.shape
        flat = fused_grid.reshape(b, t, h * w, d)
        priors = build_region_prior_grid(h, w, device=fused_grid.device, dtype=fused_grid.dtype).reshape(1, 1, self.num_predicates, h * w)
        scores = torch.einsum("btnd,pd->btpn", flat, self.queries)
        scores = scores + priors.clamp_min(1e-6).log()
        attention = entmax15(scores, dim=-1).reshape(b, t, self.num_predicates, h, w)
        tokens = torch.einsum("btphw,bthwd->btpd", attention, fused_grid)
        logits = self.presence_head(tokens).squeeze(-1)
        probs = torch.sigmoid(logits)
        confidence = 1.0 - attention.flatten(-2).var(dim=-1).clamp(0, 1)
        rel = fused_grid.new_zeros(b, max(t - 1, 0), self.num_predicates, 2)
        if t > 1:
            y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=fused_grid.device, dtype=fused_grid.dtype),
                                  torch.linspace(-1, 1, w, device=fused_grid.device, dtype=fused_grid.dtype),
                                  indexing="ij")
            pos = torch.stack([x, y], dim=-1)
            centers = torch.einsum("btphw,hwc->btpc", attention, pos)
            rel = centers[:, 1:] - centers[:, :-1]
        desc = self._descriptor(tokens, rel, confidence)
        return {
            "attention": attention,
            "tokens": tokens,
            "presence_logits": logits,
            "presence_probs": probs,
            "trajectory_confidence": confidence,
            "relative_motion": rel,
            **desc,
            "predicate_names": ACPR_PREDICATE_NAMES,
        }
