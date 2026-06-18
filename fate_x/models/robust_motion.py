from __future__ import annotations

import torch
from torch import Tensor


def weighted_geometric_median(points: Tensor, weights: Tensor, iterations: int = 2, eps: float = 1e-6) -> Tensor:
    estimate = (points * weights.unsqueeze(-1)).sum(dim=-2) / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    estimate = estimate.detach()
    for _ in range(iterations):
        dist = (points - estimate.unsqueeze(-2)).norm(dim=-1).clamp_min(eps)
        w = (weights / dist).detach()
        estimate = (points * w.unsqueeze(-1)).sum(dim=-2) / w.sum(dim=-1, keepdim=True).clamp_min(eps)
        estimate = estimate.detach()
    return estimate


def transport_displacements(matched_transport: Tensor, positions: Tensor) -> Tensor:
    pos = positions.to(matched_transport.device, matched_transport.dtype)
    mass = matched_transport.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    expected = torch.matmul(matched_transport, pos) / mass
    return expected - pos.unsqueeze(0)
