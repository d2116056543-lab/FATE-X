from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _tokens(text: str) -> List[str]:
    return [tok for tok in text.lower().replace(".", " ").replace(",", " ").split() if tok]


def sentence_cider_reward(pred: str, ref: str) -> float:
    p, r = set(_tokens(pred)), set(_tokens(ref))
    return float(len(p & r) / max(1, len(p | r)))


def sentence_meteor_reward(pred: str, ref: str) -> float:
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    overlap = sum(1 for tok in p if tok in r)
    precision = overlap / len(p)
    recall = overlap / len(r)
    return float((10 * precision * recall) / max(1e-6, recall + 9 * precision))


def hallucination_penalty(pred: str, allowed_terms: Optional[Iterable[str]] = None) -> float:
    if allowed_terms is None:
        allowed_terms = {"car", "vehicle", "road", "lane", "light", "pedestrian", "slow", "stop", "turn", "because", "ahead"}
    allowed = set(allowed_terms)
    toks = _tokens(pred)
    if not toks:
        return 1.0
    return float(sum(tok not in allowed for tok in toks) / len(toks))


def self_critical_explanation_loss(sample_logprobs: Tensor, sampled_rewards: Tensor, baseline_rewards: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    advantage = (sampled_rewards - baseline_rewards).detach()
    logp = sample_logprobs if mask is None else sample_logprobs * mask.float()
    denom = mask.float().sum().clamp_min(1.0) if mask is not None else torch.tensor(logp.numel(), device=logp.device, dtype=logp.dtype)
    return -(logp.sum(dim=-1) * advantage).sum() / denom
