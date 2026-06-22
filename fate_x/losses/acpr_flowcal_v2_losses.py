from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def gather_masked_logits(logits: Tensor, masked_pos: Tensor) -> Tensor:
    if masked_pos is None:
        return logits
    idx = masked_pos.long().unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    return logits.gather(1, idx)


def masked_language_model_loss(logits: Tensor, labels: Tensor, masked_pos: Optional[Tensor] = None) -> Tensor:
    labels = labels.to(logits.device).long()
    labels = labels.masked_fill(labels == -100, -1)
    if masked_pos is not None and masked_pos.shape == logits.shape[:2] and masked_pos.max().item() <= 1:
        selected_logits = []
        selected_labels = []
        mask = masked_pos.to(logits.device).bool()
        for b in range(logits.shape[0]):
            positions = torch.nonzero(mask[b], as_tuple=False).flatten()
            valid_labels = labels[b][labels[b] >= 0]
            n = min(int(positions.numel()), int(valid_labels.numel()))
            if n > 0:
                selected_logits.append(logits[b, positions[:n]])
                selected_labels.append(valid_labels[:n])
        if not selected_logits:
            return logits.sum() * 0.0
        return F.cross_entropy(torch.cat(selected_logits, dim=0), torch.cat(selected_labels, dim=0), ignore_index=-1)
    selected = gather_masked_logits(logits, masked_pos) if masked_pos is not None else logits
    return F.cross_entropy(selected.reshape(-1, selected.shape[-1]), labels.reshape(-1), ignore_index=-1)


def shortest_circular_delta(pred: Tensor, target: Tensor, period: float | None = None) -> Tensor:
    if period is None:
        max_abs = torch.max(torch.stack([pred.detach().abs().max(), target.detach().abs().max()])).item()
        period = 2.0 * math.pi if max_abs <= 2.0 * math.pi + 1e-4 else 360.0
    return torch.remainder(pred - target + period / 2.0, period) - period / 2.0


def normalized_control_huber(pred: Tensor, target: Tensor, stats: Optional[Dict[str, Tensor]] = None, beta: float = 1.0) -> Tensor:
    if target.ndim == 3 and target.shape[1] == 2:
        target = target.transpose(1, 2)
    if stats and "std" in stats:
        std = stats["std"].to(pred.device).view(1, 1, -1).clamp_min(1e-6)
        pred, target = pred / std, target / std
    return F.smooth_l1_loss(pred, target, beta=beta)


def control_rmse_loss(pred: Tensor, target: Tensor) -> Tensor:
    if target.ndim == 3 and target.shape[1] == 2:
        target = target.transpose(1, 2)
    return torch.sqrt(F.mse_loss(pred, target) + 1e-8)


def transport_consistency_loss(source_map: Tensor, warped_map: Tensor) -> Tensor:
    return F.l1_loss(warped_map, source_map.detach())


def lane_temporal_consistency_loss(lane_tokens: Tensor) -> Tensor:
    if lane_tokens.shape[1] < 2:
        return lane_tokens.sum() * 0.0
    return (lane_tokens[:, 1:] - lane_tokens[:, :-1]).abs().mean()


def axis_direction_weak_loss(axis_logits: Tensor, direction_logits: Tensor, targets: Dict[str, Tensor]) -> Tensor:
    loss = axis_logits.sum() * 0.0
    if "axis_targets" in targets:
        loss = loss + F.binary_cross_entropy_with_logits(axis_logits, targets["axis_targets"].float())
    if "direction_targets" in targets:
        loss = loss + F.cross_entropy(direction_logits, targets["direction_targets"].argmax(-1))
    return loss


def delta_kl_loss(base_logits: Tensor, enhanced_logits: Tensor) -> Tensor:
    base = F.log_softmax(base_logits, dim=-1)
    enh = F.softmax(enhanced_logits, dim=-1)
    return F.kl_div(base, enh, reduction="batchmean")


def parameter_anchor_loss(module: nn.Module, anchor_state: Dict[str, Tensor], weight: float = 1.0) -> Tensor:
    loss = None
    for name, param in module.named_parameters():
        if name in anchor_state:
            term = (param - anchor_state[name].to(param.device)).pow(2).mean()
            loss = term if loss is None else loss + term
    if loss is None:
        loss = next(module.parameters()).sum() * 0.0
    return loss * weight


def memory_diversity_loss(memory_values: Tensor) -> Tensor:
    x = F.normalize(memory_values, dim=-1)
    sim = torch.matmul(x, x.transpose(-1, -2))
    eye = torch.eye(sim.shape[-1], device=sim.device).view(1, sim.shape[-1], sim.shape[-1])
    return ((sim * (1 - eye)).pow(2)).mean()


def sequence_calalign_loss(base: Tensor, enhanced: Tensor, target: Tensor, alpha: float = 0.1) -> Tensor:
    return F.mse_loss(base + alpha * (enhanced - base), target)
