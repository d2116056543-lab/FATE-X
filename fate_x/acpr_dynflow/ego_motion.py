from __future__ import annotations

import torch
from torch import Tensor


def estimate_common_shift(coarse_grid: Tensor) -> Tensor:
    if coarse_grid.ndim != 5:
        raise ValueError("coarse_grid must be [B,T,H,W,D]")
    energy = coarse_grid.float().pow(2).mean(-1)
    b, t, h, w = energy.shape
    y = torch.linspace(-1, 1, h, device=energy.device)
    x = torch.linspace(-1, 1, w, device=energy.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mass = energy.sum(dim=(2, 3)).clamp_min(1e-6)
    cx = (energy * xx).sum(dim=(2, 3)) / mass
    cy = (energy * yy).sum(dim=(2, 3)) / mass
    center = torch.stack([cx, cy], dim=-1)
    shift = center[:, 1:] - center[:, :-1]
    zero = torch.zeros(b, 1, 2, device=coarse_grid.device, dtype=coarse_grid.dtype)
    return torch.cat([zero, shift], dim=1).clamp(-0.25, 0.25)

