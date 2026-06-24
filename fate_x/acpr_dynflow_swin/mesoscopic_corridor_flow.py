from __future__ import annotations

import torch
from torch import Tensor


def corridor_occupancy(corridor_mass: Tensor) -> dict[str, Tensor]:
    left, center, right = corridor_mass[..., 0], corridor_mass[..., 1], corridor_mass[..., 2]
    mobility = corridor_mass[:, 1:] - corridor_mass[:, :-1]
    return {
        "left": left,
        "center": center,
        "right": right,
        "mobility": mobility.abs().mean(dim=-1),
        "queue_pressure": center.mean(dim=-1) - 0.5 * (left.mean(dim=-1) + right.mean(dim=-1)),
        "gap_openness": torch.stack([left, right], dim=-1).max(dim=-1).values,
    }
