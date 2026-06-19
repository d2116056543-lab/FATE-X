from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LocalPartialTransport(nn.Module):
    def __init__(
        self,
        in_dim: int,
        matching_dim: int = 64,
        local_radius: int = 2,
        global_shift_radius: int = 2,
        spatial_penalty: float = 0.25,
    ) -> None:
        super().__init__()
        self.local_radius = int(local_radius)
        self.global_shift_radius = int(global_shift_radius)
        self.spatial_penalty = float(spatial_penalty)
        self.proj = nn.Linear(in_dim, matching_dim, bias=False)
        self.dustbin_logit = nn.Parameter(torch.tensor(0.0))
        self.last_memory_report: dict[str, int | bool] = {}

    def _shift_score(self, prev: Tensor, cur: Tensor, dx: int, dy: int) -> Tensor:
        shifted = torch.roll(cur, shifts=(dy, dx), dims=(1, 2))
        return (prev * shifted).sum(dim=-1).mean(dim=(1, 2))

    def _candidate_indices(self, h: int, w: int, device: torch.device) -> Tensor:
        offsets = list(itertools.product(range(-self.local_radius, self.local_radius + 1),
                                         range(-self.local_radius, self.local_radius + 1)))
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        base = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        cand = []
        for dy, dx in offsets:
            cy = (base[:, 0] + dy).clamp(0, h - 1)
            cx = (base[:, 1] + dx).clamp(0, w - 1)
            cand.append(cy * w + cx)
        return torch.stack(cand, dim=-1)

    def forward(self, fused_grid: Tensor) -> dict[str, Tensor]:
        b, t, h, w, d = fused_grid.shape
        if t < 2:
            empty = fused_grid.new_zeros(b, 0, h * w, (2 * self.local_radius + 1) ** 2 + 1)
            return {"local_transport_probs": empty, "dustbin_prob": empty[..., -1], "camera_shift": fused_grid.new_zeros(b, 0, 2)}
        projected = F.normalize(self.proj(fused_grid.float()), dim=-1)
        all_probs, all_dust, all_shift = [], [], []
        cand_idx = self._candidate_indices(h, w, fused_grid.device)
        n = h * w
        positions = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, h, device=fused_grid.device),
            torch.linspace(-1, 1, w, device=fused_grid.device),
            indexing="ij",
        ), dim=-1).reshape(n, 2)
        cand_pos = positions[cand_idx]
        for step in range(t - 1):
            prev = projected[:, step]
            cur = projected[:, step + 1]
            shifts, scores = [], []
            for dy in range(-self.global_shift_radius, self.global_shift_radius + 1):
                for dx in range(-self.global_shift_radius, self.global_shift_radius + 1):
                    shifts.append((dx, dy))
                    scores.append(self._shift_score(prev, cur, dx, dy))
            shift_scores = torch.stack(scores, dim=-1)
            shift_prob = torch.softmax(shift_scores, dim=-1)
            shift_tensor = fused_grid.new_tensor(shifts, dtype=torch.float32)
            expected = shift_prob @ shift_tensor
            prev_flat = prev.reshape(b, n, -1)
            cur_flat = cur.reshape(b, n, -1)
            cand_feat = cur_flat[:, cand_idx.reshape(-1)].reshape(b, n, cand_idx.shape[-1], -1)
            sim = (prev_flat.unsqueeze(2) * cand_feat).sum(dim=-1)
            dist = (cand_pos - positions.unsqueeze(1)).pow(2).sum(dim=-1).unsqueeze(0)
            logits = sim - self.spatial_penalty * dist
            dust = self.dustbin_logit.expand(b, n, 1)
            probs = torch.softmax(torch.cat([logits, dust], dim=-1), dim=-1)
            all_probs.append(probs.to(fused_grid.dtype))
            all_dust.append(probs[..., -1].to(fused_grid.dtype))
            all_shift.append(expected.to(fused_grid.dtype))
        out = {
            "local_transport_probs": torch.stack(all_probs, dim=1),
            "dustbin_prob": torch.stack(all_dust, dim=1),
            "camera_shift": torch.stack(all_shift, dim=1),
        }
        self.last_memory_report = {
            "dense_global_matrix": False,
            "source_tokens": n,
            "local_candidates": cand_idx.shape[-1],
        }
        return out
