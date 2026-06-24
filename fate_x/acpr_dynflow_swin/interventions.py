from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from .types import TrafficStateOutput


@dataclass(frozen=True)
class InterventionSpec:
    kind: str
    factor_index: int | None = None
    predicate_index: int | None = None
    seed: int = 0


def zero_factor(traffic: TrafficStateOutput, factor_index: int) -> TrafficStateOutput:
    tokens = traffic.factor_tokens_native.clone()
    tokens[:, :, factor_index] = 0
    aligned = traffic.lag_aligned_tokens.clone()
    aligned[:, :, factor_index] = 0
    logits = traffic.factor_logits.clone()
    logits[:, :, factor_index] = 0
    probs = traffic.factor_probs.clone()
    probs[:, :, factor_index] = 0
    evidence = traffic.evidence_maps.clone()
    evidence[:, :, factor_index] = 0
    return replace(
        traffic,
        factor_tokens_native=tokens,
        factor_logits=logits,
        factor_probs=probs,
        evidence_maps=evidence,
        lag_aligned_tokens=aligned,
    )


def temporal_reverse_frames(frames: torch.Tensor) -> torch.Tensor:
    return torch.flip(frames, dims=[1])


def run_intervention(model: Any, batch: Any, spec: InterventionSpec):
    return model(batch, intervention=spec)
