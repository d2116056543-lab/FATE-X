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
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.score(dense_tokens.to(next(self.score.parameters()).dtype)).float()
            dense_fp32 = dense_tokens.float()
            assignment = torch.softmax(logits, dim=-1).float()
            mass = assignment.sum(dim=2).float().clamp_min(1e-6)
            weighted = torch.einsum("btns,btnd->btsd", assignment, dense_fp32).float()
            native_tokens = (weighted / mass.unsqueeze(-1)).float()
            conserved_sum = torch.einsum("bts,btsd->btd", mass, native_tokens).float()
            conservation_error = (conserved_sum - dense_fp32.sum(dim=2)).abs().amax(dim=-1).float()
        if isinstance(self.project, nn.Identity):
            projected = native_tokens.to(dense_tokens.dtype)
        else:
            projected = self.project(native_tokens.to(self.project.weight.dtype))
        return SemanticTokenConsolidation(
            slot_names=self.slot_names,
            assignment=assignment,
            token_mass=mass,
            tokens=projected,
            source_provenance=assignment,
            conservation_error=conservation_error,
        )
