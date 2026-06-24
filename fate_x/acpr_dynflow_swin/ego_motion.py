from __future__ import annotations

import torch
from torch import Tensor


def estimate_common_shift(previous: Tensor, current: Tensor, max_shift: int = 2) -> Tensor:
    """Estimate one bounded image-plane shift by coarse correlation."""
    scores = []
    shifts = []
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            rolled = torch.roll(previous, shifts=(dy, dx), dims=(-3, -2))
            scores.append((rolled * current).mean(dim=(-3, -2, -1)))
            shifts.append((dx, dy))
    stacked = torch.stack(scores, dim=-1)
    best = stacked.argmax(dim=-1)
    shift_tensor = torch.tensor(shifts, device=current.device, dtype=current.dtype)
    return shift_tensor[best]
