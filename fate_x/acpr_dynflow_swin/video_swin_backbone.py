from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import SwinBackboneOutput


class ACPRDynFlowSwinBackbone(nn.Module):
    """Single-pass direct-image backbone wrapper.

    The formal runtime is designed to call the repository Video Swin path. Unit
    tests and CPU smokes use a small native-stage compatible fallback so the
    formal namespace remains importable without large assets.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.forward_count = 0
        self.stem = nn.Conv2d(3, out_dim, kernel_size=3, padding=1)
        self.mid = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.final = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, frames: Tensor) -> SwinBackboneOutput:
        if frames.ndim != 5 or frames.shape[1] != 32 or frames.shape[2] != 3:
            raise ValueError(f"frames must be [B,32,3,H,W], got {tuple(frames.shape)}")
        self.forward_count += 1
        bsz, steps, channels, height, width = frames.shape
        x = frames.reshape(bsz * steps, channels, height, width)
        x = F.gelu(self.stem(x))
        mid = F.gelu(self.mid(x))
        fin = F.gelu(self.final(mid))
        predicate = F.adaptive_avg_pool2d(mid, (4, 4)).permute(0, 2, 3, 1).reshape(bsz, steps, 4, 4, -1)
        final = F.adaptive_avg_pool2d(fin, (7, 7)).permute(0, 2, 3, 1).reshape(bsz, steps, 7, 7, -1)
        temporal = final.mean(dim=(2, 3))
        dense = final.reshape(bsz, steps * 49, -1)
        return SwinBackboneOutput(
            predicate_grid=predicate,
            final_grid=final,
            temporal_global=temporal,
            dense_final_tokens=dense,
            forward_count=self.forward_count,
        )
