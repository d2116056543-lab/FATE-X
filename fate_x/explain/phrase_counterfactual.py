from __future__ import annotations

from typing import Iterable

import torch


def recover_original_token_scores(reduced_scores: torch.Tensor, provenance: torch.Tensor | None = None) -> torch.Tensor:
    """Map reduced-token phrase evidence back to original video tokens."""
    if provenance is None:
        return reduced_scores
    if reduced_scores.ndim == 1:
        reduced_scores = reduced_scores.unsqueeze(0).expand(provenance.shape[0], -1)
    return torch.bmm(provenance.float(), reduced_scores.float().unsqueeze(-1)).squeeze(-1)


def topk_token_mask(scores: torch.Tensor, fraction: float = 0.2, largest: bool = True) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be [B,N]")
    b, n = scores.shape
    k = max(1, min(n, int(round(n * fraction))))
    idx = torch.topk(scores, k=k, dim=1, largest=largest).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def mask_video_tokens(video_tokens: torch.Tensor, token_mask: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    if video_tokens.ndim != 3 or token_mask.ndim != 2:
        raise ValueError("video_tokens must be [B,N,D] and token_mask must be [B,N]")
    out = video_tokens.clone()
    out[token_mask] = fill
    return out


def keep_only_video_tokens(video_tokens: torch.Tensor, token_mask: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    return mask_video_tokens(video_tokens, ~token_mask, fill=fill)


def score_drop(original_score: float, perturbed_score: float) -> float:
    return float(original_score) - float(perturbed_score)


def summarize_phrase_scores(records: Iterable[dict]) -> dict:
    """Aggregate phrase-level deletion/sufficiency records.

    Expected optional record fields:
      original_score, topk_masked_score, evidence_only_score, random_masked_score.
    If absent, the row is counted as lexical-only and no faithfulness metric is fabricated.
    """
    rows = list(records)
    usable = [r for r in rows if "original_score" in r and "topk_masked_score" in r]
    if not usable:
        return {"faithfulness_available": False, "phrase_records": len(rows), "usable_phrase_records": 0}
    deletion = [score_drop(r["original_score"], r["topk_masked_score"]) for r in usable]
    suff = [float(r.get("evidence_only_score", 0.0)) for r in usable if "evidence_only_score" in r]
    rand = [score_drop(r["original_score"], r["random_masked_score"]) for r in usable if "random_masked_score" in r]
    return {
        "faithfulness_available": True,
        "phrase_records": len(rows),
        "usable_phrase_records": len(usable),
        "phrase_deletion_score": float(sum(deletion) / len(deletion)),
        "phrase_sufficiency_score": float(sum(suff) / len(suff)) if suff else None,
        "random_deletion_score": float(sum(rand) / len(rand)) if rand else None,
    }