from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn

from .predicate_ontology import EXACT_32_PREDICATES


class PredicateQueryTransfer(nn.Module):
    def __init__(self, dim: int = 256, gate_init: float = 0.25):
        super().__init__()
        self.name_embedding = nn.Embedding(len(EXACT_32_PREDICATES), dim)
        self.oia_mapper = nn.Linear(dim, dim, bias=False)
        init_logit = torch.logit(torch.tensor(float(gate_init)).clamp(1e-4, 1 - 1e-4))
        self.transfer_gate_logit = nn.Parameter(torch.full((len(EXACT_32_PREDICATES),), float(init_logit)))
        self.domain_residual = nn.Parameter(torch.zeros(len(EXACT_32_PREDICATES), dim))
        self.register_buffer("oia_query", torch.zeros(len(EXACT_32_PREDICATES), dim), persistent=True)
        self.source_report: dict[str, object] = {"loaded": False, "predicate_order": EXACT_32_PREDICATES}

    def load_oia_query(self, path: str | Path, key: str = "predicate_queries") -> None:
        checkpoint_path = Path(path)
        data = checkpoint_path.read_bytes()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tensor = ckpt
        for part in key.split("."):
            tensor = tensor[part]
        if tuple(tensor.shape) != tuple(self.oia_query.shape):
            raise ValueError(f"OIA query shape mismatch: {tuple(tensor.shape)} != {tuple(self.oia_query.shape)}")
        self.oia_query.copy_(tensor)
        self.source_report = {
            "loaded": True,
            "path": str(checkpoint_path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "tensor_key": key,
            "source_shape": list(tensor.shape),
            "mapped_shape": list(self.oia_query.shape),
            "predicate_order": EXACT_32_PREDICATES,
        }

    def forward(self) -> tuple[Tensor, dict[str, object]]:
        idx = torch.arange(len(EXACT_32_PREDICATES), device=self.domain_residual.device)
        name = self.name_embedding(idx)
        gate = torch.sigmoid(self.transfer_gate_logit).unsqueeze(-1)
        query = name + gate * self.oia_mapper(self.oia_query.to(name.device)) + self.domain_residual
        return query, {**self.source_report, "transfer_gate_mean": float(gate.detach().mean().cpu())}
