from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def scaled_gradient(x: Tensor, scale: float) -> Tensor:
    return x.detach() + (x - x.detach()) * scale


def representation_pcgrad_surrogate(reason_memory: Any, semantic_loss: Tensor, control_loss: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
    surrogate = (semantic_loss * 0.0 + control_loss * 0.0)
    if hasattr(reason_memory, "values"):
        surrogate = surrogate + reason_memory.values.mean() * 0.0
    diag = {
        "semantic_loss": semantic_loss.detach(),
        "control_loss": control_loss.detach(),
        "semantic_grad_norm": semantic_loss.detach().abs(),
        "control_grad_norm": control_loss.detach().abs(),
    }
    return surrogate, diag


def apply_semantic_gradient_firewall(x: Tensor, scale: float = 1.0) -> Tensor:
    return scaled_gradient(x, scale)
