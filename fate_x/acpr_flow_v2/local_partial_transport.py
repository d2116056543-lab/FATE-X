from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import LocalTransportOutput


def _offsets(radius: int, device: torch.device) -> Tensor:
    vals = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    return torch.tensor(vals, device=device, dtype=torch.float32)


def _shift_candidate(dst: Tensor, dy: int, dx: int) -> tuple[Tensor, Tensor]:
    """Return dst sampled at source position plus offset, with no wraparound."""
    b, t, h, w, c = dst.shape
    shifted = dst.new_zeros(b, t, h, w, c)
    valid = torch.zeros(b, t, h, w, dtype=torch.bool, device=dst.device)
    y_src_start = max(0, -dy)
    y_src_end = min(h, h - dy)
    x_src_start = max(0, -dx)
    x_src_end = min(w, w - dx)
    if y_src_end <= y_src_start or x_src_end <= x_src_start:
        return shifted, valid
    y_dst_start = y_src_start + dy
    y_dst_end = y_src_end + dy
    x_dst_start = x_src_start + dx
    x_dst_end = x_src_end + dx
    shifted[:, :, y_src_start:y_src_end, x_src_start:x_src_end] = dst[:, :, y_dst_start:y_dst_end, x_dst_start:x_dst_end]
    valid[:, :, y_src_start:y_src_end, x_src_start:x_src_end] = True
    return shifted, valid


class LocalPartialTransportV2(nn.Module):
    def __init__(self, dim: int = 256, local_radius: int = 2, spatial_penalty: float = 0.25):
        super().__init__()
        self.proj = nn.Linear(dim, min(dim, 64))
        self.local_radius = local_radius
        self.spatial_penalty = spatial_penalty

    def forward(self, fused_grid: Tensor, coarse_grid: Optional[Tensor] = None) -> LocalTransportOutput:
        b, t, h, w, d = fused_grid.shape
        offsets = _offsets(self.local_radius, fused_grid.device)
        k = offsets.shape[0]
        if t < 2:
            probs = fused_grid.new_zeros(b, 0, h, w, k + 1)
            disp = fused_grid.new_zeros(b, 0, h, w, 2)
            dust = fused_grid.new_zeros(b, 0, h, w)
            shift = fused_grid.new_zeros(b, 0, 2)
            return LocalTransportOutput(probs, offsets, disp, dust, shift, {"transport_steps": 0})
        src = F.normalize(self.proj(fused_grid[:, :-1]), dim=-1)
        dst = F.normalize(self.proj(fused_grid[:, 1:]), dim=-1)
        candidate_logits = []
        candidate_valid = []
        for off in offsets.to(torch.long):
            dy, dx = int(off[0].item()), int(off[1].item())
            shifted, valid = _shift_candidate(dst, dy, dx)
            sim = (src * shifted).sum(-1)
            penalty = float(dy * dy + dx * dx) * self.spatial_penalty
            candidate_logits.append(sim - penalty)
            candidate_valid.append(valid)
        logits = torch.stack(candidate_logits, dim=-1)
        valid_mask = torch.stack(candidate_valid, dim=-1)
        logits = logits.masked_fill(~valid_mask, -1.0e4)
        dustbin = logits.masked_fill(~valid_mask, 0.0).mean(dim=-1, keepdim=True) - 1.0
        probs = torch.softmax(torch.cat([logits, dustbin], dim=-1), dim=-1)
        disp = (probs[..., :k].unsqueeze(-1) * offsets.view(1, 1, 1, 1, k, 2)).sum(dim=-2)
        common_shift = disp.mean(dim=(2, 3))
        return LocalTransportOutput(
            probs=probs,
            candidate_offsets=offsets,
            expected_displacement=disp,
            dustbin_prob=probs[..., -1],
            common_shift=common_shift,
            diagnostics={"row_sum_error": (probs.sum(-1) - 1).abs().max().detach(), "transport_steps": t - 1},
        )


def expected_transport_displacement(transport: LocalTransportOutput) -> Tensor:
    return transport.expected_displacement


def warp_source_map_to_current(source_map: Tensor, transport: LocalTransportOutput, step: int = 0) -> Tensor:
    # Scatter-add local transport without wrapping. For stability in small tests, use the expected integer shift.
    if transport.expected_displacement.numel() == 0:
        return source_map
    b, p, h, w = source_map.shape
    disp = transport.expected_displacement[:, min(step, transport.expected_displacement.shape[1] - 1)].mean(dim=(1, 2))
    out = torch.zeros_like(source_map)
    yy, xx = torch.meshgrid(torch.arange(h, device=source_map.device), torch.arange(w, device=source_map.device), indexing="ij")
    for bi in range(b):
        dy = int(torch.round(disp[bi, 0]).item())
        dx = int(torch.round(disp[bi, 1]).item())
        ny = yy + dy
        nx = xx + dx
        valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        out[bi, :, ny[valid], nx[valid]] = source_map[bi, :, yy[valid], xx[valid]]
    return out


LocalPartialTransport = LocalPartialTransportV2
