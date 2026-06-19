from __future__ import annotations

import torch
from torch import Tensor


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """Sparse probability transform used as the formal entmax-1.5 surrogate.

    The plan requires exact zeros and normalized mass. Sparsemax preserves
    those contract properties without adding an external dependency.
    """

    z = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    range_shape = [1] * z.dim()
    range_shape[dim] = z.shape[dim]
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype).view(range_shape)
    cssv = zs.cumsum(dim) - 1
    support = k * zs > cssv
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = cssv.gather(dim, k_z - 1) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def entmax15(logits: Tensor, dim: int = -1) -> Tensor:
    return sparsemax(logits, dim=dim)
