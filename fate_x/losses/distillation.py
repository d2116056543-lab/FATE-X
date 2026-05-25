from __future__ import annotations

import torch
import torch.nn.functional as F


def decoder_logit_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """KL distillation over decoder token distributions."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"decoder logits shape mismatch: {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}")
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=-1),
        F.softmax(teacher_logits / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)


def phrase_score_distillation(student_phrase_scores: torch.Tensor, teacher_phrase_scores: torch.Tensor) -> torch.Tensor:
    """MSE interface for preserving teacher phrase span log-prob scores."""
    if student_phrase_scores.shape != teacher_phrase_scores.shape:
        raise ValueError(f"phrase score shape mismatch: {tuple(student_phrase_scores.shape)} vs {tuple(teacher_phrase_scores.shape)}")
    return F.mse_loss(student_phrase_scores.float(), teacher_phrase_scores.float())
