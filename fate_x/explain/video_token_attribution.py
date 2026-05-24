from __future__ import annotations

import torch


def gradient_x_activation(video_tokens: torch.Tensor, token_gradients: torch.Tensor) -> torch.Tensor:
    """Gradient x activation attribution for video tokens."""
    if video_tokens.shape != token_gradients.shape:
        raise ValueError("video_tokens and token_gradients must have the same shape")
    return (video_tokens * token_gradients).sum(-1)


def normalize_token_scores(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mins = scores.min(dim=1, keepdim=True).values
    maxs = scores.max(dim=1, keepdim=True).values
    return (scores - mins) / (maxs - mins).clamp_min(eps)