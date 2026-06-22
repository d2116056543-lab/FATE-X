from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FrozenContextualReasonTarget(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim
        self.register_buffer("basis", torch.linspace(-1.0, 1.0, dim).view(1, dim), persistent=False)

    def encode_texts(self, texts: List[str]) -> Tensor:
        rows = []
        for text in texts:
            val = (sum(ord(c) for c in text) % 997) / 997.0
            rows.append(torch.sin(self.basis * (1.0 + val * 10.0)).squeeze(0))
        return torch.stack(rows, dim=0) if rows else self.basis.new_zeros(0, self.dim)

    def build_target(self, actions: List[str], justifications: List[str]) -> Dict[str, Tensor]:
        with torch.no_grad():
            text = [f"{a} {j}" for a, j in zip(actions, justifications)]
            target = self.encode_texts(text).detach()
            return {"reason_target": target, "target": target}


class ActionSubspaceTracker:
    def __init__(self, rank: int = 16):
        self.rank = rank
        self._items: List[Tensor] = []
        self.components: Optional[Tensor] = None

    def update(self, action_embeddings: Tensor) -> None:
        self._items.append(action_embeddings.detach().cpu())

    def finalize_epoch(self) -> None:
        if not self._items:
            return
        x = torch.cat(self._items, dim=0)
        x = x - x.mean(0, keepdim=True)
        _, _, v = torch.pca_lowrank(x, q=min(self.rank, x.shape[-1]))
        self.components = v[:, : min(self.rank, v.shape[1])].contiguous()
        self._items.clear()

    def state_dict(self) -> Dict[str, Tensor]:
        return {"components": self.components if self.components is not None else torch.empty(0)}

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        comp = state.get("components", torch.empty(0))
        self.components = comp if comp.numel() else None


def build_contextual_reason_target(actions: List[str], justifications: List[str], dim: int = 768) -> Tensor:
    return FrozenContextualReasonTarget(dim=dim).build_target(actions, justifications)["reason_target"]
