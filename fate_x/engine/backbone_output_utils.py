from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def _flatten_tensors(value) -> list[Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        tensors: list[Tensor] = []
        if "final_tokens" in value:
            tensors.extend(_flatten_tensors(value["final_tokens"]))
        elif "final" in value:
            tensors.extend(_flatten_tensors(value["final"]))
        if "stages" in value:
            tensors.extend(_flatten_tensors(value["stages"]))
        return tensors
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def unpack_flowtrace_backbone_output(backbone_out) -> tuple[Tensor, list[Tensor]]:
    """Normalize Swin output for ADAPT main path and FlowTrace stages."""
    tensors = _flatten_tensors(backbone_out)
    if not tensors:
        raise TypeError(f"Backbone output does not contain tensors: {type(backbone_out)!r}")
    return tensors[0], tensors[1:]
