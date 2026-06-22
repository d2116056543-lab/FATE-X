from __future__ import annotations

import torch
from torch import Tensor, nn

from .predicate_ontology import TRAFFIC_FACTOR_NAMES
from .dynamic_predicate_field import entmax15_like
from .types import DynamicPredicateField, PredicateCovariates, TrafficFlowState


class TrafficStateReasoner(nn.Module):
    def __init__(self, dim: int = 256, num_factors: int = 13, num_predicates: int = 32):
        super().__init__()
        self.factor_names = TRAFFIC_FACTOR_NAMES
        self.factor_queries = nn.Parameter(torch.randn(num_factors, dim) * 0.02)
        self.factor_proj = nn.Linear(dim, dim)
        self.factor_logit = nn.Linear(dim, 1)
        self.lateral = nn.Linear(dim, 1)

    def forward(self, pred: DynamicPredicateField, cov: PredicateCovariates, lane_flow: dict[str, Tensor]) -> TrafficFlowState:
        x = cov.homogenized
        score = torch.einsum("fd,btkd->btfk", self.factor_queries, x) / (x.shape[-1] ** 0.5)
        support = entmax15_like(score, dim=-1)
        factor_tokens = torch.einsum("btfk,btkd->btfd", support, x)
        if "tokens" in lane_flow:
            # Inject corridor-level mesoscopic flow so lane dynamics affect factors and downstream decisions.
            lane_context = lane_flow["tokens"].mean(2).unsqueeze(2)
            factor_tokens = factor_tokens + lane_context
        factor_tokens = self.factor_proj(factor_tokens)
        logits = self.factor_logit(factor_tokens).squeeze(-1)
        probs = torch.sigmoid(logits)
        evidence = torch.einsum("btfk,btkhw->btfhw", support, pred.evidence_maps)
        lateral_bias = torch.tanh(self.lateral(factor_tokens).mean(2))
        lineage = [{"factor": name, "source": "predicate_lane_pattern"} for name in self.factor_names]
        b, t, f, d = factor_tokens.shape
        lag_w = torch.zeros(b, 32, f, 4, device=x.device)
        lag_w[..., 0] = 1.0
        return TrafficFlowState(
            factor_names=self.factor_names,
            factor_tokens=factor_tokens,
            factor_logits=logits,
            factor_probs=probs,
            lateral_bias=lateral_bias,
            factor_to_predicate=support,
            evidence_maps=evidence,
            response_lag_weights=lag_w,
            lag_aligned_tokens=factor_tokens[:, :32] if t >= 32 else factor_tokens.repeat_interleave((32 + t - 1) // t, dim=1)[:, :32],
            lineage=lineage,
        )

