from __future__ import annotations

from typing import Any

import torch


def unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return a plain state dict from a checkpoint object."""
    if hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, dict):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    return checkpoint


def filter_compatible_state_dict(
    model: torch.nn.Module,
    checkpoint: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, object]]]:
    """Keep only checkpoint tensors that exist in the model with identical shape.

    PyTorch's ``strict=False`` still raises on shape mismatches. ADAPT's released
    basemodel has a one-signal sensor head, while the official multitask script
    uses two signals (course, speed), so the incompatible head must be skipped.
    """
    model_state = model.state_dict()
    raw_state = unwrap_state_dict(checkpoint)
    filtered: dict[str, torch.Tensor] = {}
    skipped: dict[str, dict[str, object]] = {}
    for key, value in raw_state.items():
        if key not in model_state:
            skipped[key] = {"reason": "missing_in_model"}
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped[key] = {
                "reason": "shape_mismatch",
                "checkpoint_shape": tuple(value.shape),
                "model_shape": tuple(model_state[key].shape),
            }
            continue
        filtered[key] = value
    return filtered, skipped
