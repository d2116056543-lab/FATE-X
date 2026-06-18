from __future__ import annotations

import torch
from torch import Tensor


def entmax15(logits: Tensor, dim: int = -1, n_iter: int = 50, eps: float = 1e-6) -> Tensor:
    """Numerically stable entmax-1.5 via threshold bisection.

    This self-contained implementation is intentionally small and dependency-free.
    It returns sparse probabilities that sum to one on ``dim``.
    """
    x = logits.transpose(dim, -1)
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    x = x - x.max(dim=-1, keepdim=True).values
    tau_lo = x.min(dim=-1, keepdim=True).values - 1.0
    tau_hi = x.max(dim=-1, keepdim=True).values
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.relu((x - tau) / 2.0).pow(2)
        mass = p.sum(dim=-1, keepdim=True)
        tau_lo = torch.where(mass > 1.0, tau, tau_lo)
        tau_hi = torch.where(mass <= 1.0, tau, tau_hi)
    tau = tau_hi
    p = torch.relu((x - tau) / 2.0).pow(2)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = p.reshape(orig_shape).transpose(dim, -1)
    return p
