from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ContradictionAwareTemporalHardPair(nn.Module):
    def __init__(self, queue_size: int = 4096, margin: float = 0.2):
        super().__init__()
        self.queue_size = queue_size
        self.margin = margin
        self.register_buffer("queue", torch.empty(0), persistent=False)

    def mine(self, embeddings: Tensor, labels: Tensor) -> Dict[str, Tensor]:
        sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        diff = labels.unsqueeze(1) != labels.unsqueeze(0)
        return {"similarity": sim, "negative_mask": diff}

    def enqueue(self, embeddings: Tensor) -> None:
        if not self.training:
            return
        emb = embeddings.detach().cpu()
        self.queue = torch.cat([self.queue.cpu(), emb], dim=0)[-self.queue_size :]

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        mined = self.mine(embeddings, labels)
        neg = mined["similarity"][mined["negative_mask"]]
        if neg.numel() == 0:
            return embeddings.sum() * 0.0
        return F.relu(neg - self.margin).mean()


def temporal_hardpair_margin_loss(embeddings: Tensor, labels: Tensor, margin: float = 0.2) -> Tensor:
    return ContradictionAwareTemporalHardPair(margin=margin)(embeddings, labels)
