from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def weighted_loss_total(
    raw_losses: dict[str, Tensor],
    weights: dict[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    missing = [name for name, weight in weights.items() if float(weight) != 0.0 and name not in raw_losses]
    if missing:
        raise KeyError(f"configured nonzero losses are missing from the formal graph: {missing}")
    weighted = {
        name: value * float(weights.get(name, 0.0))
        for name, value in raw_losses.items()
    }
    if not weighted:
        raise ValueError("formal loss graph is empty")
    return torch.stack([value.reshape(()) for value in weighted.values()]).sum(), weighted


def pattern_targets_from_predicates(probabilities: Tensor, threshold: float = 0.02) -> Tensor:
    activity = probabilities.mean(dim=-1)
    delta = torch.zeros_like(activity)
    delta[:, 1:] = activity[:, 1:] - activity[:, :-1]
    previous = torch.zeros_like(delta)
    previous[:, 1:] = delta[:, :-1]
    oscillating = delta.mul(previous).lt(0) & delta.abs().gt(threshold) & previous.abs().gt(threshold)
    targets = torch.zeros_like(activity, dtype=torch.long)
    targets = torch.where(delta.gt(threshold), torch.ones_like(targets), targets)
    targets = torch.where(delta.lt(-threshold), torch.full_like(targets, 2), targets)
    return torch.where(oscillating, torch.full_like(targets, 3), targets)


def traffic_targets_from_predicates(
    probabilities: Tensor,
    pattern_targets: Tensor | None = None,
) -> Tensor:
    pattern_targets = (
        pattern_targets_from_predicates(probabilities)
        if pattern_targets is None
        else pattern_targets
    )

    def group(*indices: int) -> Tensor:
        return probabilities[..., list(indices)].amax(dim=-1)

    phase = F.one_hot(pattern_targets, num_classes=4).to(probabilities.dtype)
    return torch.stack(
        [
            group(2, 14, 18),
            group(4, 6),
            group(17),
            group(5, 15, 16),
            phase[..., 1],
            phase[..., 0],
            phase[..., 2],
            phase[..., 3],
            group(0, 1, 2, 3),
            group(4, 5, 6, 7),
            group(8, 9, 10, 11, 12, 13, 14, 15, 22),
            group(19, 20, 21, 22, 29),
            group(23, 24, 25, 26, 27, 28, 31),
        ],
        dim=-1,
    ).clamp(0, 1)


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
    with torch.cuda.amp.autocast(enabled=False):
        return F.binary_cross_entropy(
            gate.float().clamp(1e-6, 1.0 - 1e-6),
            target.detach().float().clamp(0, 1),
        )


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
