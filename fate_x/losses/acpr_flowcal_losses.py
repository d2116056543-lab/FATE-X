from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def partial_label_bce(logits: Tensor, positive: Tensor, contradiction: Tensor, reliability: Tensor) -> Tensor:
    target = positive
    weight = torch.where((positive + contradiction) > 0, torch.ones_like(reliability), reliability)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    contradiction_loss = F.binary_cross_entropy_with_logits(logits, 1.0 - contradiction, reduction="none")
    mixed = torch.where(contradiction > 0, contradiction_loss, loss)
    return (mixed * weight).sum() / weight.sum().clamp_min(1.0)


def masked_l2_loss(pred: Tensor, target: Tensor, invalid_value: float = -1.0) -> Tensor:
    mask = target.ne(invalid_value)
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp_min(1)


def memory_diversity_loss(memory: Tensor) -> Tensor:
    norm = F.normalize(memory, dim=-1)
    sim = torch.matmul(norm, norm.transpose(-1, -2))
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
    return sim.masked_fill(eye, 0.0).pow(2).mean()


def build_acpr_loss_components(action_text: Tensor, explanation_text: Tensor, control: Tensor,
                               predicate_pu: Tensor, flow_pu: Tensor, reason_semantic: Tensor,
                               hardpair_budgeted: Tensor, memory_diversity: Tensor,
                               weights: dict[str, float]) -> dict[str, Tensor]:
    components = {
        "action_text": action_text * weights.get("action_text", 1.0),
        "explanation_text": explanation_text * weights.get("explanation_text", 1.0),
        "control": control * weights.get("control", 0.05),
        "predicate_pu": predicate_pu * weights.get("predicate_pu", 0.05),
        "flow_pu": flow_pu * weights.get("flow_pu", 0.03),
        "reason_semantic": reason_semantic * weights.get("reason_semantic", 0.05),
        "hardpair_budgeted": hardpair_budgeted,
        "memory_diversity": memory_diversity * weights.get("memory_diversity", 0.001),
    }
    components["total"] = sum(components.values())
    return components
