from __future__ import annotations

import torch
from torch import Tensor, nn

from .activations import entmax15
from .region_priors import FLOW_FACTOR_NAMES, build_factor_support


class FlowFactorComposer(nn.Module):
    def __init__(self, state_dim: int = 256, use_grammar_prior: bool = True) -> None:
        super().__init__()
        self.num_factors = len(FLOW_FACTOR_NAMES)
        self.queries = nn.Parameter(torch.randn(self.num_factors, state_dim) * 0.02)
        self.logit = nn.Linear(state_dim, 1)
        support, contradiction = build_factor_support()
        self.register_buffer("factor_predicate_support", support)
        self.register_buffer("factor_predicate_contradiction", contradiction)
        self.use_grammar_prior = use_grammar_prior

    def forward(self, predicate_descriptor: Tensor, predicate_attention: Tensor) -> dict[str, Tensor]:
        scores = torch.einsum("bpd,kd->bkp", predicate_descriptor, self.queries)
        if self.use_grammar_prior:
            scores = scores + self.factor_predicate_support.to(scores) - self.factor_predicate_contradiction.to(scores)
        attn = entmax15(scores, dim=-1)
        tokens = torch.einsum("bkp,bpd->bkd", attn, predicate_descriptor)
        logits = self.logit(tokens).squeeze(-1)
        probs = torch.sigmoid(logits)
        maps = torch.einsum("bkp,btphw->btkhw", attn, predicate_attention)
        return {
            "flow_tokens": tokens,
            "flow_logits": logits,
            "flow_probs": probs,
            "flow_to_predicate_attention": attn,
            "flow_evidence_maps": maps,
            "flow_factor_names": FLOW_FACTOR_NAMES,
        }
