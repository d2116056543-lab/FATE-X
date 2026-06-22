from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import LaneFlowFieldOutput, PredicateTrajectory

REGION_NAMES = ("left", "center", "right")


def build_soft_corridor_masks(batch: int, time: int, height: int, width: int, device: torch.device) -> Tensor:
    xs = torch.linspace(0, 1, width, device=device).view(1, 1, 1, 1, width)
    centers = torch.tensor([0.25, 0.5, 0.75], device=device).view(1, 1, 3, 1, 1)
    masks = torch.exp(-((xs - centers) ** 2) / 0.035)
    masks = masks.expand(batch, time, 3, height, width)
    return masks / masks.sum(dim=2, keepdim=True).clamp_min(1e-6)


def refine_masks_with_drivable_predicates(masks: Tensor, predicates: PredicateTrajectory) -> Tensor:
    evidence = predicates.attention[:, :, :3].sum(2, keepdim=True)
    refined = masks * (1.0 + evidence)
    return refined / refined.sum(dim=2, keepdim=True).clamp_min(1e-6)


def aggregate_region_statistics(predicates: PredicateTrajectory, masks: Tensor) -> Dict[str, Tensor]:
    pred_mass = predicates.attention.sum(2)
    occupancy = (masks * pred_mass.unsqueeze(2)).sum(dim=(-1, -2))
    motion = predicates.relative_motion.mean(2) if predicates.relative_motion.numel() else occupancy.new_zeros(occupancy.shape[0], max(occupancy.shape[1] - 1, 0), 2)
    motion = F.pad(motion, (0, 0, 1, 0))
    motion = motion.unsqueeze(2).expand(-1, -1, 3, -1)
    coherence = torch.exp(-motion.norm(dim=-1))
    stopped = torch.sigmoid(3.0 * (0.2 - motion.norm(dim=-1)))
    queue = occupancy * stopped
    return {"occupancy": occupancy, "relative_motion": motion, "motion_coherence": coherence, "stopped_tendency": stopped, "queue_pressure": queue}


def temporal_encode_regions(stats: Dict[str, Tensor], dim: int, projector: nn.Module) -> Tensor:
    x = torch.stack([stats["occupancy"], stats["motion_coherence"], stats["stopped_tendency"], stats["queue_pressure"]], dim=-1)
    return projector(x)


class PredicateConditionedLaneFlowField(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.projector = nn.Linear(4, dim)

    def forward(self, predicates: PredicateTrajectory, fused_grid: Tensor) -> LaneFlowFieldOutput:
        b, t, h, w, d = fused_grid.shape
        masks = build_soft_corridor_masks(b, t, h, w, fused_grid.device)
        masks = refine_masks_with_drivable_predicates(masks, predicates)
        stats = aggregate_region_statistics(predicates, masks)
        temporal = temporal_encode_regions(stats, d, self.projector)
        descriptor = temporal.mean(1)
        return LaneFlowFieldOutput(
            region_names=REGION_NAMES,
            soft_masks=masks,
            occupancy=stats["occupancy"],
            relative_motion=stats["relative_motion"],
            motion_coherence=stats["motion_coherence"],
            stopped_tendency=stats["stopped_tendency"],
            queue_pressure=stats["queue_pressure"],
            temporal_tokens=temporal,
            descriptor=descriptor,
        )


LaneFlowField = PredicateConditionedLaneFlowField
