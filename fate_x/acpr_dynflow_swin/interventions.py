from __future__ import annotations

from dataclasses import replace

import torch

from .types import TrafficStateOutput


def zero_factor(traffic: TrafficStateOutput, factor_index: int) -> TrafficStateOutput:
    tokens = traffic.factor_tokens_native.clone()
    tokens[:, :, factor_index] = 0
    probs = traffic.factor_probs.clone()
    probs[:, :, factor_index] = 0
    return replace(traffic, factor_tokens_native=tokens, factor_probs=probs)


def temporal_reverse_frames(frames: torch.Tensor) -> torch.Tensor:
    return torch.flip(frames, dims=[1])
