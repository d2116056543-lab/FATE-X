from __future__ import annotations

import torch
from torch import Tensor, nn

from .predicate_ontology import TRAFFIC_FACTOR_NAMES
from .types import TrafficStateOutput


class PatternLagTrafficReasoner(nn.Module):
    pattern_names = ("stable", "forming", "releasing", "oscillating")

    def __init__(self, predicate_dim: int, factor_dim: int = 64, factor_count: int = 13):
        super().__init__()
        self.factor_count = factor_count
        self.factor_query = nn.Parameter(torch.randn(factor_count, predicate_dim) * 0.02)
        self.factor_proj = nn.Linear(predicate_dim, factor_dim)
        self.pattern_head = nn.Linear(predicate_dim, len(self.pattern_names))
        self.factor_head = nn.Linear(factor_dim, 1)
        self.lag_logits = nn.Parameter(torch.zeros(4))

    def forward(self, predicate_tokens: Tensor, evidence_maps: Tensor, corridor_mass: Tensor, target_steps: int = 32) -> TrafficStateOutput:
        bsz, steps, predicates, dim = predicate_tokens.shape
        attn = torch.softmax(torch.einsum("fd,btkd->btkf", self.factor_query, predicate_tokens), dim=2)
        native = torch.einsum("btkf,btkd->btfd", attn, predicate_tokens)
        native = self.factor_proj(native)
        factor_logits = self.factor_head(native).squeeze(-1)
        factor_probs = torch.sigmoid(factor_logits)
        pattern_logits = self.pattern_head(predicate_tokens.mean(dim=2))
        pattern_probs = torch.softmax(pattern_logits, dim=-1)
        lag_weights = torch.softmax(self.lag_logits, dim=0).view(1, 1, 1, 4).expand(bsz, target_steps, self.factor_count, -1)
        repeat = (target_steps + steps - 1) // steps
        aligned = native.repeat_interleave(repeat, dim=1)[:, :target_steps]
        evidence = torch.einsum("btkf,btkhw->btfhw", attn, evidence_maps)
        factor_to_corridor = torch.einsum("btkf,btkc->btfc", attn, corridor_mass)
        return TrafficStateOutput(
            factor_names=TRAFFIC_FACTOR_NAMES,
            factor_tokens_native=native,
            factor_logits=factor_logits,
            factor_probs=factor_probs,
            lateral_bias=factor_to_corridor[..., 2:3].mean(dim=2) - factor_to_corridor[..., 0:1].mean(dim=2),
            pattern_logits=pattern_logits,
            pattern_probs=pattern_probs,
            factor_to_predicate=attn.permute(0, 1, 3, 2),
            factor_to_corridor=factor_to_corridor,
            evidence_maps=evidence,
            lag_weights=lag_weights,
            lag_aligned_tokens=aligned,
            lineage=[{"source": "dynamic_predicates", "target": name} for name in TRAFFIC_FACTOR_NAMES],
        )
