from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class InterventionSpec:
    kind: str
    index: int | None = None
    seed: int = 0
    prefix_frames: int | None = None


def apply_intervention_to_fused_grid(fused_grid: Tensor, evidence_map: Tensor | None, spec: InterventionSpec) -> Tensor:
    if spec.kind == "none":
        return fused_grid
    if spec.kind == "temporal_reverse":
        return fused_grid.flip(1)
    if spec.kind == "temporal_shuffle":
        g = torch.Generator(device=fused_grid.device)
        g.manual_seed(spec.seed)
        idx = torch.randperm(fused_grid.shape[1], generator=g, device=fused_grid.device)
        return fused_grid[:, idx]
    if spec.kind == "prefix_length":
        n = int(spec.prefix_frames or fused_grid.shape[1])
        out = fused_grid.clone()
        if n < fused_grid.shape[1]:
            out[:, n:] = 0
        return out
    if spec.kind in {"evidence_tube_off", "random_equal_mass"}:
        if evidence_map is None:
            raise ValueError("evidence_map is required")
        mask = evidence_map.to(fused_grid.dtype).unsqueeze(-1)
        if spec.kind == "random_equal_mass":
            flat = mask.flatten(1, -2)
            k = int((flat > 0).sum(dim=1).float().mean().item())
            rand = torch.zeros_like(flat)
            g = torch.Generator(device=fused_grid.device)
            g.manual_seed(spec.seed)
            for b in range(flat.shape[0]):
                idx = torch.randperm(flat.shape[1], generator=g, device=fused_grid.device)[:k]
                rand[b, idx] = 1
            mask = rand.reshape_as(mask)
        mean = fused_grid.mean(dim=(2, 3), keepdim=True)
        return fused_grid * (1 - mask) + mean * mask
    raise ValueError(f"unknown intervention {spec.kind}")
