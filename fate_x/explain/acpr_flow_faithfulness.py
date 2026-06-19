from __future__ import annotations

import torch
from torch import Tensor


def intervention_delta(base_logits: Tensor, counterfactual_logits: Tensor) -> Tensor:
    return (base_logits.softmax(-1) - counterfactual_logits.softmax(-1)).abs().sum(dim=-1)


def evidence_beats_random(evidence_delta: Tensor, random_delta: Tensor) -> bool:
    return bool(torch.nanmean(evidence_delta.float()) > torch.nanmean(random_delta.float()))
