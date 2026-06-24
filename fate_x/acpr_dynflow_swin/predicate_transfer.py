from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import Tensor, nn

from .predicate_ontology import EXACT_32_PREDICATES


class PredicateQueryTransfer(nn.Module):
    def __init__(self, dim: int = 256, gate_init: float = 0.25, name_features: Tensor | None = None):
        super().__init__()
        if name_features is None:
            name_features = torch.eye(len(EXACT_32_PREDICATES))
        self.register_buffer("name_features", name_features.float(), persistent=True)
        self.name_mapper = nn.Linear(name_features.shape[-1], dim, bias=False)
        self.oia_mapper: nn.Linear | None = None
        init_logit = torch.logit(torch.tensor(float(gate_init)).clamp(1e-4, 1 - 1e-4))
        self.transfer_gate_logit = nn.Parameter(torch.full((len(EXACT_32_PREDICATES),), float(init_logit)))
        self.domain_residual = nn.Parameter(torch.zeros(len(EXACT_32_PREDICATES), dim))
        self.register_buffer("oia_query", torch.zeros(len(EXACT_32_PREDICATES), 0), persistent=True)
        self.source_report: dict[str, object] = {"loaded": False, "predicate_order": EXACT_32_PREDICATES}

    def load_oia_query(self, path: str | Path, key: str = "predicate_queries") -> None:
        checkpoint_path = resolve_runtime_path(path)
        data = checkpoint_path.read_bytes()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tensor = _find_checkpoint_tensor(ckpt, key)
        if tensor.ndim != 2 or tensor.shape[0] != len(EXACT_32_PREDICATES):
            raise ValueError(f"OIA query must be [32,D], got {tuple(tensor.shape)}")
        self.oia_query = tensor.detach().float().clone()
        self._materialize_oia_mapper(int(tensor.shape[-1]))
        self.source_report = {
            "loaded": True,
            "path": str(checkpoint_path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "tensor_key": key,
            "source_shape": list(tensor.shape),
            "mapped_shape": [len(EXACT_32_PREDICATES), self.domain_residual.shape[-1]],
            "predicate_order": EXACT_32_PREDICATES,
        }

    def _materialize_oia_mapper(self, source_dim: int) -> None:
        if self.oia_mapper is None or self.oia_mapper.in_features != source_dim:
            self.oia_mapper = nn.Linear(source_dim, self.domain_residual.shape[-1], bias=False).to(
                self.domain_residual.device
            )

    def forward(self) -> tuple[Tensor, dict[str, object]]:
        name = self.name_mapper(self.name_features.to(self.domain_residual.device))
        gate = torch.sigmoid(self.transfer_gate_logit).unsqueeze(-1)
        if self.oia_query.shape[-1]:
            self._materialize_oia_mapper(int(self.oia_query.shape[-1]))
            transferred = self.oia_mapper(self.oia_query.to(name.device))
        else:
            transferred = torch.zeros_like(name)
        query = name + gate * transferred + self.domain_residual
        return query, {**self.source_report, "transfer_gate_mean": float(gate.detach().mean().cpu())}


def resolve_runtime_path(path: str | Path) -> Path:
    text = str(path).replace("\\", "/")
    if os.name != "nt" and len(text) > 2 and text[1:3] == ":/":
        text = f"/mnt/{text[0].lower()}/{text[3:]}"
    return Path(text)


def _find_checkpoint_tensor(checkpoint, key: str) -> Tensor:
    current = checkpoint
    try:
        for part in key.split("."):
            current = current[part]
        if torch.is_tensor(current):
            return current
    except (KeyError, TypeError):
        pass
    candidates = []
    if isinstance(checkpoint, dict):
        candidates.append(checkpoint)
        candidates.extend(
            checkpoint[name]
            for name in ("model", "state_dict", "module", "net")
            if isinstance(checkpoint.get(name), dict)
        )
    key_without_model = key[len("model.") :] if key.startswith("model.") else key
    suffixes = (key, key_without_model, "predicate_head.predicate_queries")
    for state in candidates:
        for state_key, value in state.items():
            clean = str(state_key)
            clean = clean[len("module.") :] if clean.startswith("module.") else clean
            if torch.is_tensor(value) and any(clean == suffix or clean.endswith("." + suffix) for suffix in suffixes):
                return value
    raise KeyError(f"{key} not found in checkpoint")


@torch.no_grad()
def load_bert_name_features(bert_dir: str | Path) -> Tensor:
    from transformers import BertModel, BertTokenizer

    bert_dir = resolve_runtime_path(bert_dir)
    tokenizer = BertTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
    model = BertModel.from_pretrained(str(bert_dir), local_files_only=True)
    model.eval()
    encoded = tokenizer(
        [name.replace("_", " ") for name in EXACT_32_PREDICATES],
        padding=True,
        return_tensors="pt",
    )
    output = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1)
    return (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
