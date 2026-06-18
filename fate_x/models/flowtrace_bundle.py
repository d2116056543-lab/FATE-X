from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class FlowTraceBundle:
    dense_video_tokens: Tensor
    fine_grid: Tensor
    coarse_grid: Tensor
    fused_grid: Tensor
    transport_matrices: Tensor
    transport_confidence: Tensor
    dustbin_mass: Tensor
    track_attention: Tensor
    track_tokens: Tensor
    camera_motion: Tensor
    relative_motion: Tensor
    track_unmatched_mass: Tensor
    state_tokens_temporal: Tensor
    state_memory: Tensor
    state_scores: Tensor
    state_track_weights: Tensor
    state_evidence_maps: Tensor
    reason_state: Tensor
    reason_state_distribution: Tensor
    token_state_routing: Tensor | None = None
    pmt_delta: Tensor | None = None
    pmt_gate: Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_diagnostics(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.diagnostics)
        for name in (
            "dense_video_tokens",
            "fine_grid",
            "coarse_grid",
            "fused_grid",
            "transport_matrices",
            "track_attention",
            "track_tokens",
            "state_memory",
            "state_evidence_maps",
            "reason_state",
        ):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                out[f"{name}_shape"] = list(value.shape)
                out[f"{name}_finite"] = bool(torch.isfinite(value).all().detach().cpu())
        return out
