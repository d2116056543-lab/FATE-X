from __future__ import annotations

from torch import Tensor
import torch.nn.functional as F


def segment_caption_ce(logits: Tensor, labels: Tensor, mask: Tensor | None = None) -> Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
    if mask is None:
        return loss.mean()
    flat_mask = mask.reshape(-1).to(loss.dtype)
    return (loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
