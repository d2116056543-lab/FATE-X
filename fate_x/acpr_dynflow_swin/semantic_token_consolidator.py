from __future__ import annotations

import torch
from torch import Tensor, nn

from .types import SemanticTokenConsolidation


class SemanticTokenConsolidator(nn.Module):
    slot_names = ("global_action", "longitudinal", "left_lateral", "right_lateral", "residual_context")

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.score = nn.Linear(input_dim, len(self.slot_names))
        self.project = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, dense_tokens: Tensor) -> SemanticTokenConsolidation:
        logits = self.score(dense_tokens)
        assignment = torch.softmax(logits, dim=-1)
        mass = assignment.sum(dim=2).clamp_min(1e-6)
        weighted = torch.einsum("btns,btnd->btsd", assignment, dense_tokens)
        native_tokens = weighted / mass.unsqueeze(-1)
        projected = self.project(native_tokens)
        recon = torch.einsum("btns,btsd->btnd", assignment, native_tokens)
        conservation_error = (recon.sum(dim=2) - dense_tokens.sum(dim=2)).abs().amax(dim=-1)
        return SemanticTokenConsolidation(
            slot_names=self.slot_names,
            assignment=assignment,
            token_mass=mass,
            tokens=projected,
            source_provenance=assignment,
            conservation_error=conservation_error,
        )
