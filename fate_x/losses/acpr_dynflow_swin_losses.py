from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def signal_specific_normalized_huber(pred: Tensor, target: Tensor, signal_index: int) -> Tensor:
    return F.smooth_l1_loss(pred[..., signal_index], target[..., signal_index])


def target_delta_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def nnpu_loss(logits: Tensor, positive: Tensor, reliable_negative: Tensor, prior: Tensor) -> Tensor:
    pos = F.binary_cross_entropy_with_logits(logits, positive, reduction="none") * positive
    neg = F.binary_cross_entropy_with_logits(-logits, reliable_negative, reduction="none") * reliable_negative
    risk = pos.mean() + torch.clamp(neg.mean(), min=0.0) * prior.mean().clamp_min(1e-6)
    return risk


def pattern_semantic_loss(logits: Tensor, target: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def traffic_state_semantic_loss(logits: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def residual_target_loss(contribution: Tensor, target_residual: Tensor) -> Tensor:
    return F.smooth_l1_loss(contribution.sum(dim=2), target_residual.detach())


def benefit_gate_loss(gate: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy(gate, target.detach().clamp(0, 1))


def non_degradation_hinge(full_error: Tensor, global_error: Tensor, margin: float = 0.01) -> Tensor:
    return torch.relu(full_error - global_error + margin).mean()


def contribution_alignment_js(attn: Tensor, contrib: Tensor) -> Tensor:
    attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)
    contrib = contrib.abs()
    contrib = contrib / contrib.sum(-1, keepdim=True).clamp_min(1e-6)
    m = 0.5 * (attn + contrib)
    return 0.5 * (attn * (attn.clamp_min(1e-6).log() - m.clamp_min(1e-6).log())).sum(-1).mean() + 0.5 * (contrib * (contrib.clamp_min(1e-6).log() - m.clamp_min(1e-6).log())).sum(-1).mean()


def group_sparsity(x: Tensor) -> Tensor:
    return x.abs().mean()


def temporal_smoothness(x: Tensor) -> Tensor:
    return (x[:, 1:] - x[:, :-1]).pow(2).mean()
