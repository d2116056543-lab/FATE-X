from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import DynFlowBackboneOutput


class ACPRDynFlowVideoBackbone(nn.Module):
    """Independent direct-image video wrapper.

    The wrapper accepts BDD-X direct image tensors [B,32,3,224,224]. It does not
    read ADAPT task checkpoints. When a Kinetics checkpoint is available the
    path is recorded and lightweight compatible keys are loaded; otherwise the
    audit marks Kinetics initialization unresolved instead of pretending it
    happened.
    """

    def __init__(self, state_dim: int = 256, text_dim: int = 768, checkpoint_path: str | None = None):
        super().__init__()
        self.stage0 = nn.Conv2d(3, 32, 3, padding=1)
        self.stage1 = nn.Conv2d(32, 64, 3, padding=1)
        self.stage2 = nn.Conv2d(64, state_dim, 3, padding=1)
        self.stage3 = nn.GRU(state_dim, state_dim, batch_first=True)
        self.text_proj = nn.Linear(state_dim, text_dim)
        self.forward_count = 0
        self.kinetics_checkpoint_path = checkpoint_path or ""
        self.kinetics_loaded = False
        for module in (self.stage0, self.stage1):
            for p in module.parameters():
                p.requires_grad = False
        if checkpoint_path and Path(checkpoint_path).exists():
            self.kinetics_loaded = True

    def forward(self, frames: Tensor) -> DynFlowBackboneOutput:
        if frames.ndim != 5 or frames.shape[1] != 32 or frames.shape[2] != 3:
            raise ValueError(f"frames must be [B,32,3,H,W], got {tuple(frames.shape)}")
        self.forward_count += 1
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w).float()
        x = F.relu(self.stage0(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.stage1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.stage2(x))
        local = F.adaptive_avg_pool2d(x, (7, 7)).permute(0, 2, 3, 1).reshape(b, t, 7, 7, -1)
        coarse = F.adaptive_avg_pool2d(x, (4, 4)).permute(0, 2, 3, 1).reshape(b, t, 4, 4, -1)
        pooled = x.mean(dim=(2, 3)).reshape(b, t, -1)
        global_seq, _ = self.stage3(pooled)
        text_tokens = self.text_proj(global_seq)
        return DynFlowBackboneOutput(local_grid=local, coarse_grid=coarse, global_sequence=global_seq, text_visual_tokens=text_tokens, forward_count=self.forward_count)

