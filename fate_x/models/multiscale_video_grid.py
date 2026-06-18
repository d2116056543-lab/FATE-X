from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MultiScaleBackboneOutput:
    final_tokens: Tensor
    stages: list[Tensor]


class MultiScaleVideoGrid(nn.Module):
    def __init__(self, fine_dim: int, coarse_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.fine_proj = nn.Linear(fine_dim, out_dim)
        self.coarse_proj = nn.Linear(coarse_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    @staticmethod
    def to_bt_hw_c(stage: Tensor) -> Tensor:
        if stage.dim() != 5:
            raise ValueError(f"Expected 5D stage tensor, got {tuple(stage.shape)}")
        # Swin returns [B,C,T,H,W]. Already-converted grids are [B,T,H,W,C].
        # In Video Swin, channel is the largest dimension for normal BDD-X settings.
        if stage.shape[1] >= max(stage.shape[2], stage.shape[3], stage.shape[4]):
            return stage.permute(0, 2, 3, 4, 1).contiguous()
        return stage

    def forward(self, fine_stage: Tensor, coarse_stage: Tensor, final_tokens: Tensor | None = None) -> dict[str, Tensor]:
        fine = self.to_bt_hw_c(fine_stage)
        coarse = self.to_bt_hw_c(coarse_stage)
        fine_p = self.fine_proj(fine)
        coarse_p = self.coarse_proj(coarse)
        bt = fine_p.shape[0] * fine_p.shape[1]
        coarse_up = F.interpolate(
            coarse_p.reshape(bt, coarse_p.shape[2], coarse_p.shape[3], coarse_p.shape[4]).permute(0, 3, 1, 2),
            size=fine_p.shape[2:4],
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape_as(fine_p)
        fused = self.norm(fine_p + coarse_up)
        dense = final_tokens
        if dense is None:
            dense = coarse.reshape(coarse.shape[0], -1, coarse.shape[-1])
        return {"fine_grid": fine_p, "coarse_grid": coarse_p, "fused_grid": fused, "dense_tokens": dense}
