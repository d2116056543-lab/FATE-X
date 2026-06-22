from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .predicate_ontology import EXACT_32_PREDICATES


def checkpoint_sha256(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PredicateQueryInitializer(nn.Module):
    def __init__(self, dim: int = 256, checkpoint_path: str | None = None):
        super().__init__()
        self.names = EXACT_32_PREDICATES
        gen = torch.Generator().manual_seed(20260622)
        base = torch.randn(len(self.names), dim, generator=gen) * 0.02
        name = torch.stack([self._name_embedding(n, dim) for n in self.names])
        self.register_buffer("oia_prior", base)
        self.register_buffer("name_prior", name)
        self.oia_mapper = nn.Linear(dim, dim, bias=False)
        self.name_mapper = nn.Linear(dim, dim, bias=False)
        self.residual = nn.Parameter(torch.zeros(len(self.names), dim))
        self.checkpoint_path = checkpoint_path or ""
        self.checkpoint_sha256 = checkpoint_sha256(checkpoint_path) if checkpoint_path else None

    @staticmethod
    def _name_embedding(name: str, dim: int) -> Tensor:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        vals = torch.tensor([digest[i % len(digest)] for i in range(dim)], dtype=torch.float32)
        return (vals / 255.0 - 0.5) * 0.1

    def forward(self) -> tuple[Tensor, dict[str, Any]]:
        q_oia = self.oia_mapper(self.oia_prior.detach())
        q_name = self.name_mapper(self.name_prior.detach())
        q = q_oia + q_name + self.residual
        report = {
            "names": list(self.names),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "oia_norm": float(q_oia.detach().norm().cpu()),
            "name_norm": float(q_name.detach().norm().cpu()),
            "residual_norm": float(self.residual.detach().norm().cpu()),
        }
        return q, report

