from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def contribution_js_loss(attention: Tensor, contributions: Tensor) -> Tensor:
    p = torch.softmax(attention, dim=-1)
    q = contributions.detach().abs().sum(-1)
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-6)
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div((p + 1e-8).log(), m, reduction="batchmean") + F.kl_div((q + 1e-8).log(), m, reduction="batchmean"))

