from __future__ import annotations

import torch
from torch import Tensor, nn

from .dynamic_traffic_state_composer import DynamicTrafficStateComposer
from .flowtrace_bundle import FlowTraceBundle
from .multiscale_video_grid import MultiScaleVideoGrid
from .predicted_reason_state import PredictedReasonState
from .transported_evidence_tracks import TransportedEvidenceTracks


class FlowTracePMTModel(nn.Module):
    def __init__(self, fine_dim: int, coarse_dim: int, dense_dim: int, state_dim: int = 256,
                 num_tracks: int = 12, num_states: int = 8) -> None:
        super().__init__()
        self.grid = MultiScaleVideoGrid(fine_dim, coarse_dim, state_dim)
        self.tracks = TransportedEvidenceTracks(state_dim, state_dim=state_dim, num_tracks=num_tracks)
        self.composer = DynamicTrafficStateComposer(state_dim=state_dim, num_states=num_states, num_tracks=num_tracks)
        self.reason = PredictedReasonState(state_dim)
        self.dense_proj = nn.Linear(dense_dim, dense_dim)

    def forward(self, dense_video_tokens: Tensor, fine_stage: Tensor, coarse_stage: Tensor) -> FlowTraceBundle:
        grids = self.grid(fine_stage, coarse_stage, final_tokens=dense_video_tokens)
        tr = self.tracks(grids["fused_grid"])
        comp = self.composer(
            tr["track_tokens"],
            tr["track_attention"],
            tr["relative_motion"],
            tr["track_unmatched_mass"],
        )
        reason = self.reason(comp["state_memory"])
        dust = tr["transport_matrices"][..., -1].mean(dim=-1)
        return FlowTraceBundle(
            dense_video_tokens=dense_video_tokens,
            fine_grid=grids["fine_grid"],
            coarse_grid=grids["coarse_grid"],
            fused_grid=grids["fused_grid"],
            transport_matrices=tr["transport_matrices"],
            transport_confidence=tr["track_confidence"],
            dustbin_mass=dust,
            track_attention=tr["track_attention"],
            track_tokens=tr["track_tokens"],
            camera_motion=tr["camera_motion"],
            relative_motion=tr["relative_motion"],
            track_unmatched_mass=tr["track_unmatched_mass"],
            state_tokens_temporal=comp["state_tokens_temporal"],
            state_memory=comp["state_memory"],
            state_scores=comp["state_scores"],
            state_track_weights=comp["state_track_weights"],
            state_evidence_maps=comp["state_evidence_maps"],
            reason_state=reason["reason_state"],
            reason_state_distribution=reason["reason_state_distribution"],
            diagnostics={"flowtrace_enabled": True},
        )
