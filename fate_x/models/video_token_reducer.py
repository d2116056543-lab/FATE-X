from __future__ import annotations

import warnings

import torch
from torch import nn
import torch.nn.functional as F


class VideoTokenReducer(nn.Module):
    """Keep+merge token reducer with provenance for ADAPT video tokens.

    The reducer is intentionally order-preserving: selected tokens are returned in
    original video-token order, not score order. This keeps temporal/spatial
    structure readable for the VL transformer and prevents accidental damage to
    downstream control heads when a reducer is explicitly enabled for them.
    """

    def __init__(
        self,
        dim: int,
        keep_ratio: float = 0.5,
        num_summary_tokens: int = 1,
        min_tokens: int = 128,
        mode: str = "topk_merge",
        temporal_tokens: int | None = None,
        spatial_tokens_per_frame: int | None = None,
        min_tokens_per_frame: int = 1,
        summary_mode: str = "cluster",
    ) -> None:
        super().__init__()
        if mode not in {"none", "merge", "topk_merge", "per_frame_topk_merge"}:
            raise ValueError(f"Unsupported reducer mode: {mode}")
        if summary_mode not in {"global_mean", "cluster", "per_frame_cluster"}:
            raise ValueError(f"Unsupported summary mode: {summary_mode}")
        self.mode = mode
        self.keep_ratio = float(keep_ratio)
        self.num_summary_tokens = int(num_summary_tokens)
        self.min_tokens = int(min_tokens)
        self.temporal_tokens = temporal_tokens
        self.spatial_tokens_per_frame = spatial_tokens_per_frame
        self.min_tokens_per_frame = int(min_tokens_per_frame)
        self.summary_mode = summary_mode
        hidden = max(dim // 2, 1)
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @staticmethod
    def identity_provenance(batch: int, tokens: int, device=None, dtype=None) -> torch.Tensor:
        eye = torch.eye(tokens, device=device, dtype=dtype or torch.float32)
        return eye.unsqueeze(0).expand(batch, tokens, tokens).contiguous()

    def _keep_count(self, n: int) -> int:
        if self.mode == "none":
            return n
        if self.mode == "merge":
            return min(max(int(self.min_tokens), 1), n)
        keep = max(int(self.min_tokens), int(round(n * self.keep_ratio)))
        return min(max(keep, 1), n)

    def _global_keep_indices(self, scores: torch.Tensor, keep: int) -> torch.Tensor:
        keep_idx = torch.topk(scores, k=keep, dim=1).indices
        return torch.sort(keep_idx, dim=1).values

    def _per_frame_keep_indices(self, scores: torch.Tensor) -> torch.Tensor | None:
        b, n = scores.shape
        if not self.temporal_tokens or not self.spatial_tokens_per_frame:
            return None
        t = int(self.temporal_tokens)
        s = int(self.spatial_tokens_per_frame)
        if t <= 0 or s <= 0 or t * s != n:
            return None
        per_frame_keep = max(int(self.min_tokens_per_frame), int(round(s * self.keep_ratio)))
        per_frame_keep = min(max(per_frame_keep, 1), s)
        frame_scores = scores.view(b, t, s)
        keep_idx = torch.topk(frame_scores, k=per_frame_keep, dim=2).indices
        keep_idx = torch.sort(keep_idx, dim=2).values
        frame_offsets = (torch.arange(t, device=scores.device).view(1, t, 1) * s)
        return (keep_idx + frame_offsets).reshape(b, t * per_frame_keep)

    @staticmethod
    def _cluster_assign(rest_tokens: torch.Tensor, num_clusters: int) -> torch.Tensor:
        n = rest_tokens.shape[0]
        if n == 0:
            return torch.empty(0, device=rest_tokens.device, dtype=torch.long)
        if num_clusters <= 1 or n <= 1:
            return torch.zeros(n, device=rest_tokens.device, dtype=torch.long)
        k = min(num_clusters, n)
        normed = F.normalize(rest_tokens.float(), dim=-1)
        centers = [0]
        dist = 1.0 - (normed @ normed[0].view(-1, 1)).squeeze(1)
        for _ in range(1, k):
            idx = int(torch.argmax(dist).item())
            centers.append(idx)
            new_dist = 1.0 - (normed @ normed[idx].view(-1, 1)).squeeze(1)
            dist = torch.minimum(dist, new_dist)
        center_tensor = torch.tensor(centers, device=rest_tokens.device, dtype=torch.long)
        sim = normed @ normed[center_tensor].transpose(0, 1)
        return torch.argmax(sim, dim=1)

    def _summary_tokens(self, sample_tokens: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        r, d = sample_tokens.shape
        if r == 0 or m <= 0:
            return sample_tokens.new_zeros(0, d), sample_tokens.new_zeros(r, 0)
        if self.summary_mode == "global_mean" or m == 1:
            summary = sample_tokens.mean(0, keepdim=True).expand(m, d).contiguous()
            weights = sample_tokens.new_full((r, m), 1.0 / float(m))
            return summary, weights
        assign = self._cluster_assign(sample_tokens, m)
        summaries = []
        weights = sample_tokens.new_zeros(r, m)
        for ci in range(m):
            mask = assign == ci
            if bool(mask.any()):
                summaries.append(sample_tokens[mask].mean(0))
                weights[mask, ci] = 1.0
            else:
                summaries.append(sample_tokens.mean(0))
        return torch.stack(summaries, dim=0), weights

    def forward(self, video_tokens: torch.Tensor, attention_proxy: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict]:
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must be [B,N,D]")
        b, n, d = video_tokens.shape
        if attention_proxy is not None:
            base_scores = attention_proxy.to(video_tokens.device).float() + video_tokens.norm(dim=-1) * 0.01
        else:
            base_scores = self.score(video_tokens).squeeze(-1) + video_tokens.norm(dim=-1) * 0.01
        if self.mode == "none":
            prov = self.identity_provenance(b, n, video_tokens.device, video_tokens.dtype)
            return {
                "tokens": video_tokens,
                "provenance": prov,
                "scores": base_scores,
                "stats": {
                    "original_tokens": n,
                    "kept_tokens": n,
                    "summary_tokens": 0,
                    "reduced_tokens": n,
                    "summary_mode": "none",
                    "order_preserved": True,
                },
            }

        if self.mode == "per_frame_topk_merge":
            keep_idx = self._per_frame_keep_indices(base_scores)
            if keep_idx is None:
                warnings.warn("per_frame_topk_merge layout unavailable; falling back to global topk_merge", RuntimeWarning)
                keep_idx = self._global_keep_indices(base_scores, self._keep_count(n))
                per_frame = False
            else:
                per_frame = True
        else:
            keep_idx = self._global_keep_indices(base_scores, self._keep_count(n))
            per_frame = False

        keep = keep_idx.shape[1]
        m = min(max(int(self.num_summary_tokens), 0), max(n - keep, 0))
        batch = torch.arange(b, device=video_tokens.device).unsqueeze(1)
        kept = video_tokens.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, d))
        prov = torch.zeros(b, n, keep + m, device=video_tokens.device, dtype=video_tokens.dtype)
        prov[batch, keep_idx, torch.arange(keep, device=video_tokens.device).view(1, -1)] = 1.0

        if m > 0:
            mask = torch.ones(b, n, device=video_tokens.device, dtype=torch.bool)
            mask[batch, keep_idx] = False
            sample_summaries = []
            for bi in range(b):
                residual_idx = torch.nonzero(mask[bi], as_tuple=False).squeeze(1)
                summary, weights = self._summary_tokens(video_tokens[bi, residual_idx], m)
                sample_summaries.append(summary)
                prov[bi, residual_idx, keep:] = weights
            reduced = torch.cat([kept, torch.stack(sample_summaries, dim=0)], dim=1)
        else:
            reduced = kept

        return {
            "tokens": reduced,
            "provenance": prov,
            "scores": base_scores,
            "stats": {
                "original_tokens": n,
                "kept_tokens": keep,
                "summary_tokens": m,
                "reduced_tokens": int(reduced.shape[1]),
                "summary_mode": self.summary_mode if m > 0 else "none",
                "order_preserved": True,
                "per_frame_topk": bool(per_frame),
            },
        }
