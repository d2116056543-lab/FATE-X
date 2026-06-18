from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ReasonAnchorArtifacts:
    anchors: Tensor
    train_sample_ids: list[str]
    nearest_justifications: list[list[str]]
    fingerprint: dict


class ReasonStateAnchorBank(nn.Module):
    def __init__(self, anchors: Tensor) -> None:
        super().__init__()
        self.register_buffer("anchors", F.normalize(anchors.float(), dim=-1))

    def forward(self, state_memory: Tensor) -> Tensor:
        pooled = F.normalize(state_memory.mean(dim=1), dim=-1)
        return torch.softmax(torch.matmul(pooled, self.anchors.t()), dim=-1)

    @classmethod
    def load(cls, path: str | Path) -> "ReasonStateAnchorBank":
        data = torch.load(path, map_location="cpu")
        return cls(data["anchors"])


def ridge_residualize(action_embeddings: Tensor, reason_embeddings: Tensor, ridge_lambda: float = 1e-3,
                      rho: float = 1.0) -> Tensor:
    a = action_embeddings.float()
    r = reason_embeddings.float()
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    w = torch.linalg.solve(a.t().matmul(a) + ridge_lambda * eye, a.t().matmul(r))
    return F.normalize(r - rho * a.matmul(w), dim=-1)


def spherical_kmeans(x: Tensor, k: int = 8, iterations: int = 50, seed: int = 42) -> tuple[Tensor, Tensor]:
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x = F.normalize(x.float(), dim=-1)
    perm = torch.randperm(x.shape[0], generator=gen, device=x.device)
    centers = x[perm[:k]].clone()
    for _ in range(iterations):
        labels = torch.matmul(x, centers.t()).argmax(dim=-1)
        new = []
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                new.append(F.normalize(x[mask].mean(dim=0), dim=0))
            else:
                new.append(centers[idx])
        centers = torch.stack(new, dim=0)
    labels = torch.matmul(x, centers.t()).argmax(dim=-1)
    return centers, labels


def save_anchor_artifacts(path: str | Path, artifacts: ReasonAnchorArtifacts) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "anchors": artifacts.anchors.detach().cpu(),
        "train_sample_ids": artifacts.train_sample_ids,
        "nearest_justifications": artifacts.nearest_justifications,
        "fingerprint": artifacts.fingerprint,
    }, path)
    path.with_suffix(".json").write_text(json.dumps({
        "train_sample_ids": artifacts.train_sample_ids,
        "nearest_justifications": artifacts.nearest_justifications,
        "fingerprint": artifacts.fingerprint,
    }, indent=2), encoding="utf-8")
