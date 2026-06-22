from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .ego_motion import estimate_common_shift
from .predicate_ontology import EXACT_32_PREDICATES
from .types import DynamicPredicateField


def entmax15_like(logits: Tensor, dim: int = -1) -> Tensor:
    probs = torch.softmax(logits, dim=dim)
    cutoff = probs.mean(dim=dim, keepdim=True) * 0.25
    probs = torch.where(probs >= cutoff, probs, torch.zeros_like(probs))
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-6)


class ACPRDynamicPredicateField(nn.Module):
    def __init__(self, dim: int = 256, num_predicates: int = 32):
        super().__init__()
        self.names = EXACT_32_PREDICATES
        self.query_gru = nn.GRUCell(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.presence = nn.Linear(dim, 1)
        self.query_norm = nn.LayerNorm(dim)
        self.num_predicates = num_predicates

    def forward(self, local_grid: Tensor, coarse_grid: Tensor, initial_queries: Tensor) -> DynamicPredicateField:
        b, t, h, w, d = local_grid.shape
        keys = self.key(local_grid)
        vals = self.value(local_grid)
        q = initial_queries.unsqueeze(0).expand(b, -1, -1)
        query_states = []
        tokens = []
        evidences = []
        logits = []
        for step in range(t):
            grid = keys[:, step].reshape(b, h * w, d)
            score = torch.einsum("bkd,bnd->bkn", q, grid) / (d ** 0.5)
            attn = entmax15_like(score, dim=-1)
            val = vals[:, step].reshape(b, h * w, d)
            tok = torch.einsum("bkn,bnd->bkd", attn, val)
            q = self.query_norm(self.query_gru(tok.reshape(b * self.num_predicates, d), q.reshape(b * self.num_predicates, d))).reshape(b, self.num_predicates, d)
            query_states.append(q)
            tokens.append(tok)
            evidences.append(attn.reshape(b, self.num_predicates, h, w))
            logits.append(self.presence(tok).squeeze(-1))
        query_states_t = torch.stack(query_states, dim=1)
        tokens_t = torch.stack(tokens, dim=1)
        evidence = torch.stack(evidences, dim=1)
        logit = torch.stack(logits, dim=1)
        prob = torch.sigmoid(logit)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=local_grid.device), torch.linspace(-1, 1, w, device=local_grid.device), indexing="ij")
        centroid = torch.stack([(evidence * xx).sum((-1, -2)), (evidence * yy).sum((-1, -2))], dim=-1)
        delta = centroid[:, 1:] - centroid[:, :-1]
        shift = estimate_common_shift(coarse_grid)[:, 1:].unsqueeze(2)
        rel = delta - shift
        lane_left = evidence[..., :, : max(1, w // 3)].sum((-1, -2))
        lane_center = evidence[..., :, max(1, w // 3) : max(2, 2 * w // 3)].sum((-1, -2))
        lane_right = evidence[..., :, max(2, 2 * w // 3) :].sum((-1, -2))
        lane = torch.stack([lane_left, lane_center, lane_right], dim=-1)
        conf = (prob * evidence.flatten(-2).amax(-1)).clamp(0, 1)
        return DynamicPredicateField(
            names=self.names,
            logits=logit,
            probabilities=prob,
            tokens=tokens_t,
            evidence_maps=evidence,
            confidence=conf,
            centroid=centroid,
            relative_centroid_motion=rel,
            lane_mass=lane / lane.sum(-1, keepdim=True).clamp_min(1e-6),
            query_states=query_states_t,
        )

