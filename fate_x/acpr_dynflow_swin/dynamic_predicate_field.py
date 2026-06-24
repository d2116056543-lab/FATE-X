from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .predicate_ontology import EXACT_32_PREDICATES
from .types import DynamicPredicateField


class DynamicPredicateFieldBuilder(nn.Module):
    def __init__(self, dim: int, predicate_count: int = 32):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(predicate_count, dim) * 0.02)
        self.transfer_gate = nn.Parameter(torch.full((predicate_count,), -1.0986123))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, predicate_grid: Tensor) -> DynamicPredicateField:
        bsz, steps, height, width, dim = predicate_grid.shape
        flat = predicate_grid.reshape(bsz, steps, height * width, dim)
        keys = self.key(flat)
        values = self.value(flat)
        scores = torch.einsum("kd,bthd->btkh", self.queries, keys) / max(dim, 1) ** 0.5
        evidence = F.softmax(scores, dim=-1)
        tokens = torch.einsum("btkh,bthd->btkd", evidence, values)
        logits = tokens.mean(-1)
        probs = torch.sigmoid(logits)
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=predicate_grid.device),
            torch.linspace(0, 1, width, device=predicate_grid.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).reshape(height * width, 2)
        centroid = torch.einsum("btkh,hd->btkd", evidence, coords)
        rel = centroid[:, 1:] - centroid[:, :-1]
        corridor = torch.stack(
            [
                evidence[..., : height * width // 3].sum(-1),
                evidence[..., height * width // 3 : 2 * height * width // 3].sum(-1),
                evidence[..., 2 * height * width // 3 :].sum(-1),
            ],
            dim=-1,
        )
        return DynamicPredicateField(
            names=EXACT_32_PREDICATES,
            query_states=self.queries.unsqueeze(0).unsqueeze(0).expand(bsz, steps, -1, -1),
            logits=logits,
            probabilities=probs,
            tokens=tokens,
            evidence_maps=evidence.reshape(bsz, steps, len(EXACT_32_PREDICATES), height, width),
            confidence=(probs - 0.5).abs() * 2.0,
            centroid=centroid,
            relative_motion=rel,
            corridor_mass=corridor,
            transfer_gate=torch.sigmoid(self.transfer_gate),
        )
