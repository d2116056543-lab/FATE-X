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
        self.pattern_branches = nn.ModuleList(
            nn.Conv1d(predicate_dim, predicate_dim, kernel_size=3, padding=dilation, dilation=dilation)
            for dilation in (1, 2, 4)
        )
        self.pattern_fusion = nn.Linear(predicate_dim * 3, predicate_dim)
        self.factor_proj = nn.Linear(predicate_dim, factor_dim)
        self.pattern_to_factor = nn.Linear(predicate_dim, factor_dim)
        self.pattern_head = nn.Linear(predicate_dim, len(self.pattern_names))
        self.factor_head = nn.Linear(factor_dim, 1)
        self.lag_logits = nn.Parameter(torch.zeros(factor_count, 4))

    @staticmethod
    def _causal_shift(tokens: Tensor, lag: int) -> Tensor:
        if lag == 0:
            return tokens
        padding = tokens.new_zeros(tokens.shape[0], lag, tokens.shape[2], tokens.shape[3])
        return torch.cat([padding, tokens[:, :-lag]], dim=1)

    def forward(self, predicate_tokens: Tensor, evidence_maps: Tensor, corridor_mass: Tensor, target_steps: int = 32) -> TrafficStateOutput:
        bsz, steps, predicates, dim = predicate_tokens.shape
        pooled = predicate_tokens.mean(dim=2)
        temporal = pooled.transpose(1, 2)
        pattern_features = torch.cat([torch.tanh(branch(temporal)) for branch in self.pattern_branches], dim=1)
        pattern_context = self.pattern_fusion(pattern_features.transpose(1, 2))
        attn = torch.softmax(torch.einsum("fd,btkd->btkf", self.factor_query, predicate_tokens), dim=2)
        native = torch.einsum("btkf,btkd->btfd", attn, predicate_tokens)
        native = self.factor_proj(native) + self.pattern_to_factor(pattern_context).unsqueeze(2)
        factor_logits = self.factor_head(native).squeeze(-1)
        factor_probs = torch.sigmoid(factor_logits)
        pattern_logits = self.pattern_head(pattern_context)
        pattern_probs = torch.softmax(pattern_logits, dim=-1)
        repeat = (target_steps + steps - 1) // steps
        target_native = native.repeat_interleave(repeat, dim=1)[:, :target_steps]
        lag_weights = torch.softmax(self.lag_logits, dim=-1).view(1, 1, self.factor_count, 4)
        lag_weights = lag_weights.expand(bsz, target_steps, -1, -1)
        lagged = torch.stack([self._causal_shift(target_native, lag) for lag in range(4)], dim=-2)
        aligned = (lagged * lag_weights.unsqueeze(-1)).sum(dim=-2)
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
