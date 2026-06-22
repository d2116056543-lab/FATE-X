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


def _find_oia_predicate_state(ckpt: Any) -> tuple[Tensor | None, dict[str, Tensor], str]:
    states: list[tuple[str, dict[str, Tensor]]] = []
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "module", "net"):
            if isinstance(ckpt.get(key), dict):
                states.append((key, ckpt[key]))
        states.append(("<top>", ckpt))
    for source, sd in states:
        q = None
        extras: dict[str, Tensor] = {}
        for key, value in sd.items():
            clean = key.replace("module.", "")
            if clean == "predicate_head.predicate_queries" and torch.is_tensor(value):
                q = value.float()
            elif clean.startswith("predicate_head.") and torch.is_tensor(value):
                extras[clean] = value.detach().cpu()
        if q is not None:
            return q, extras, source
    return None, {}, ""


class PredicateQueryInitializer(nn.Module):
    def __init__(self, dim: int = 256, checkpoint_path: str | None = None):
        super().__init__()
        self.names = EXACT_32_PREDICATES
        self.checkpoint_path = checkpoint_path or ""
        self.checkpoint_sha256 = checkpoint_sha256(checkpoint_path) if checkpoint_path else None
        self.oia_loaded = False
        self.oia_load_error = ""
        self.oia_source = ""
        self.oia_source_dim = dim
        self.oia_extra_keys: list[str] = []

        gen = torch.Generator().manual_seed(20260622)
        oia_prior = torch.randn(len(self.names), dim, generator=gen) * 0.02
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                loaded, extras, source = _find_oia_predicate_state(ckpt)
                if loaded is None:
                    raise KeyError("predicate_head.predicate_queries not found in checkpoint")
                if loaded.shape[0] < len(self.names):
                    raise ValueError(f"checkpoint has {loaded.shape[0]} queries, expected at least {len(self.names)}")
                oia_prior = loaded[: len(self.names)].contiguous()
                self.oia_loaded = True
                self.oia_source = source
                self.oia_source_dim = int(oia_prior.shape[1])
                self.oia_extra_keys = sorted(extras.keys())
            except Exception as exc:
                self.oia_load_error = f"{type(exc).__name__}: {exc}"

        name = torch.stack([self._name_embedding(n, dim) for n in self.names])
        self.register_buffer("oia_prior", oia_prior.float())
        self.register_buffer("name_prior", name)
        self.oia_mapper = nn.Linear(int(oia_prior.shape[1]), dim, bias=False)
        self.name_mapper = nn.Linear(dim, dim, bias=False)
        self.residual = nn.Parameter(torch.zeros(len(self.names), dim))

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
            "oia_loaded": self.oia_loaded,
            "oia_load_error": self.oia_load_error,
            "oia_source": self.oia_source,
            "oia_source_dim": self.oia_source_dim,
            "oia_extra_keys": self.oia_extra_keys,
            "oia_prior_shape": list(self.oia_prior.shape),
            "oia_norm": float(q_oia.detach().norm().cpu()),
            "name_norm": float(q_name.detach().norm().cpu()),
            "residual_norm": float(self.residual.detach().norm().cpu()),
        }
        return q, report
