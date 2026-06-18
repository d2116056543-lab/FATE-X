from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .entmax15 import entmax15
from .robust_motion import transport_displacements, weighted_geometric_median
from .sinkhorn_transport import LogSinkhornTransport


def make_grid_positions(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device, dtype=dtype),
                          torch.linspace(-1, 1, w, device=device, dtype=dtype),
                          indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)


class TransportedEvidenceTracks(nn.Module):
    def __init__(self, in_dim: int, state_dim: int = 256, num_tracks: int = 12,
                 matching_dim: int = 64, beta_max: float = 2.0) -> None:
        super().__init__()
        self.num_tracks = num_tracks
        self.beta_max = beta_max
        self.token_proj = nn.Linear(in_dim, state_dim)
        self.track_queries = nn.Parameter(torch.randn(num_tracks, state_dim) * 0.02)
        self.pos_bias = nn.Linear(2, num_tracks, bias=False)
        self.transport = LogSinkhornTransport(state_dim, matching_dim=matching_dim)

    def forward(self, fused_grid: Tensor) -> dict[str, Tensor]:
        b, t, h, w, _ = fused_grid.shape
        n = h * w
        x = self.token_proj(fused_grid).reshape(b, t, n, -1)
        positions = make_grid_positions(h, w, fused_grid.device, x.dtype)
        pos_bias = self.pos_bias(positions).transpose(0, 1)  # [L,N]
        attentions = []
        tokens = []
        confs = []
        unmatched = []
        transports = []
        camera = []
        rel_motion = []
        prev_attn = None
        for ti in range(t):
            logits = torch.einsum("bnd,ld->bln", x[:, ti], self.track_queries) / (x.shape[-1] ** 0.5)
            logits = logits + pos_bias.unsqueeze(0)
            if ti > 0 and prev_attn is not None:
                tr = self.transport(x[:, ti - 1], x[:, ti], positions)
                matched = tr["matched_transport"]
                transports.append(tr["transport"])
                conf = tr["match_confidence"].mean(dim=-1)
                dust = tr["source_dustbin_mass"].mean(dim=-1)
                beta = self.beta_max * conf * (1.0 - dust).clamp(0, 1)
                prior = torch.matmul(prev_attn, matched)
                logits = logits + beta[:, None, None] * torch.log(prior.clamp_min(1e-6))
                disp = transport_displacements(matched, positions)
                weights = matched.sum(dim=-1).detach()
                cam = weighted_geometric_median(disp, weights)
                camera.append(cam)
                rel = torch.einsum("bln,bnd->bld", prev_attn, disp - cam[:, None, :])
                rel_motion.append(rel)
                confs.append(conf[:, None].expand(-1, self.num_tracks))
                unmatched.append(dust[:, None].expand(-1, self.num_tracks))
            else:
                eye = torch.eye(n + 1, device=x.device, dtype=x.dtype).unsqueeze(0).expand(b, -1, -1)
                transports.append(eye)
                confs.append(torch.ones(b, self.num_tracks, device=x.device, dtype=x.dtype))
                unmatched.append(torch.zeros(b, self.num_tracks, device=x.device, dtype=x.dtype))
            attn = entmax15(logits, dim=-1)
            tok = torch.einsum("bln,bnd->bld", attn, x[:, ti])
            attentions.append(attn.reshape(b, self.num_tracks, h, w))
            tokens.append(tok)
            prev_attn = attn
        if not camera:
            camera_t = torch.zeros(b, 0, 2, device=x.device, dtype=x.dtype)
            rel_t = torch.zeros(b, 0, self.num_tracks, 2, device=x.device, dtype=x.dtype)
        else:
            camera_t = torch.stack(camera, dim=1)
            rel_t = torch.stack(rel_motion, dim=1)
        return {
            "track_attention": torch.stack(attentions, dim=1),
            "track_tokens": torch.stack(tokens, dim=1),
            "track_confidence": torch.stack(confs, dim=1),
            "track_unmatched_mass": torch.stack(unmatched, dim=1),
            "transport_matrices": torch.stack(transports, dim=1),
            "camera_motion": camera_t,
            "relative_motion": rel_t,
        }
