from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _grid_positions(num_tokens: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    side = int(num_tokens ** 0.5)
    if side * side == num_tokens:
        y, x = torch.meshgrid(torch.linspace(-1, 1, side, device=device, dtype=dtype),
                              torch.linspace(-1, 1, side, device=device, dtype=dtype),
                              indexing="ij")
        return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
    return torch.stack([
        torch.linspace(-1, 1, num_tokens, device=device, dtype=dtype),
        torch.zeros(num_tokens, device=device, dtype=dtype),
    ], dim=-1)


class LogSinkhornTransport(nn.Module):
    def __init__(self, in_dim: int, matching_dim: int = 64, spatial_penalty: float = 0.25,
                 sinkhorn_iterations: int = 4, epsilon: float = 0.07) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, matching_dim, bias=False)
        self.spatial_penalty = spatial_penalty
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.dustbin = nn.Parameter(torch.tensor(0.0))

    def forward(self, previous: Tensor, current: Tensor, positions: Tensor | None = None) -> dict[str, Tensor]:
        with torch.cuda.amp.autocast(enabled=False):
            prev = previous.float()
            cur = current.float()
            b, n, _ = prev.shape
            if positions is None:
                positions = _grid_positions(n, prev.device, prev.dtype)
            pos = positions.float().to(prev.device)
            prev_m = F.normalize(self.proj(prev), dim=-1)
            cur_m = F.normalize(self.proj(cur), dim=-1)
            sim = torch.matmul(prev_m, cur_m.transpose(-1, -2))
            dist = torch.cdist(pos, pos).pow(2).unsqueeze(0)
            score = (sim - self.spatial_penalty * dist) / max(self.epsilon, 1e-6)
            dust_row = self.dustbin.expand(b, n, 1).float()
            dust_col = self.dustbin.expand(b, 1, n + 1).float()
            log_alpha = torch.cat([torch.cat([score, dust_row], dim=-1), dust_col], dim=1)
            log_p = log_alpha
            for _ in range(self.sinkhorn_iterations):
                log_p = log_p - torch.logsumexp(log_p, dim=2, keepdim=True)
                log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            transport = log_p.exp()
            matched = transport[:, :n, :n]
            match_conf = matched.sum(dim=-1)
            source_dust = transport[:, :n, n]
            target_dust = transport[:, n, :n]
            row_res = (transport.sum(dim=-1) - 1.0).abs()
            col_res = (transport.sum(dim=-2) - 1.0).abs()
            return {
                "transport": transport.to(previous.dtype),
                "matched_transport": matched.to(previous.dtype),
                "match_confidence": match_conf.to(previous.dtype),
                "source_dustbin_mass": source_dust.to(previous.dtype),
                "target_dustbin_mass": target_dust.to(previous.dtype),
                "max_row_residual": row_res.max(),
                "max_col_residual": col_res.max(),
            }
