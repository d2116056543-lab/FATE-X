from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def normalized_huber(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.numel() == 0:
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred, target)


def first_difference_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    return normalized_huber(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], mask[:, 1:] if mask is not None else None)

