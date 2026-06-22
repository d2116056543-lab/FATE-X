from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PrefixFuturePredictor(nn.Module):
    def __init__(self, dim: int = 256, target_frames: int = 8):
        super().__init__()
        self.target_frames = target_frames
        self.head = nn.Linear(dim, dim)

    def forward(self, precomputed_grids: Tensor, prefix_frames: int = 24) -> Tensor:
        prefix = precomputed_grids[:, :prefix_frames].mean(dim=(1, 2, 3))
        pred = self.head(prefix).unsqueeze(1).expand(-1, self.target_frames, -1)
        return pred


def build_prefix_bundle_from_precomputed_grids(precomputed_grids: Tensor, prefix_frames: int = 24, target_frames: int = 8) -> Dict[str, Tensor]:
    prefix = precomputed_grids[:, :prefix_frames]
    future = precomputed_grids[:, prefix_frames : prefix_frames + target_frames]
    return {"prefix_grid": prefix, "target_grid": future, "prefix": prefix, "future": future}


PrefixFutureHead = PrefixFuturePredictor
