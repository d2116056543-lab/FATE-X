from __future__ import annotations

import torch
from torch import nn


class VideoTokenReducer(nn.Module):
    """Keep+merge token reducer with provenance for ADAPT video tokens."""

    def __init__(
        self,
        dim: int,
        keep_ratio: float = 0.5,
        num_summary_tokens: int = 64,
        min_tokens: int = 128,
        mode: str = "topk_merge",
    ) -> None:
        super().__init__()
        if mode not in {"none", "merge", "topk_merge"}:
            raise ValueError(f"Unsupported reducer mode: {mode}")
        self.mode = mode
        self.keep_ratio = keep_ratio
        self.num_summary_tokens = num_summary_tokens
        self.min_tokens = min_tokens
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

    def forward(self, video_tokens: torch.Tensor, attention_proxy: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict]:
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must be [B,N,D]")
        b, n, d = video_tokens.shape
        if self.mode == "none":
            prov = self.identity_provenance(b, n, video_tokens.device, video_tokens.dtype)
            return {"tokens": video_tokens, "provenance": prov, "scores": video_tokens.norm(dim=-1), "stats": {"original_tokens": n, "kept_tokens": n, "summary_tokens": 0, "reduced_tokens": n}}
        scores = self.score(video_tokens).squeeze(-1) + video_tokens.norm(dim=-1) * 0.01
        if attention_proxy is not None:
            scores = scores + attention_proxy.to(scores.device).float()
        keep = self._keep_count(n)
        m = min(max(int(self.num_summary_tokens), 0), max(n - keep, 0))
        order = torch.argsort(scores, dim=1, descending=True)
        keep_idx = order[:, :keep]
        batch = torch.arange(b, device=video_tokens.device).unsqueeze(1)
        kept = video_tokens.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, d))
        prov = torch.zeros(b, n, keep + m, device=video_tokens.device, dtype=video_tokens.dtype)
        prov[batch, keep_idx, torch.arange(keep, device=video_tokens.device).view(1, -1)] = 1.0
        if m > 0:
            mask = torch.ones(b, n, device=video_tokens.device, dtype=torch.bool)
            mask[batch, keep_idx] = False
            rest = video_tokens.masked_fill(~mask.unsqueeze(-1), 0.0)
            summary = rest.sum(1, keepdim=True) / mask.sum(1).clamp_min(1).to(video_tokens.dtype).view(b, 1, 1)
            summary = summary.expand(-1, m, -1).contiguous()
            prov[:, :, keep:] = mask.to(video_tokens.dtype).unsqueeze(-1) / float(m)
            reduced = torch.cat([kept, summary], 1)
        else:
            reduced = kept
        return {"tokens": reduced, "provenance": prov, "scores": scores, "stats": {"original_tokens": n, "kept_tokens": keep, "summary_tokens": m, "reduced_tokens": reduced.shape[1]}}