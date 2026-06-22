from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import InterventionSpecV2


class FlowCalV2InterventionEngine:
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model

    def rerun_from_visual(self, batch: Any, spec: InterventionSpecV2) -> Any:
        if self.model is None:
            return replace(batch) if is_dataclass(batch) else batch
        return self.model(batch, intervention=spec)

    def rerun_from_predicates(self, bundle: Any, spec: InterventionSpecV2) -> Any:
        bundle = replace(bundle) if is_dataclass(bundle) else bundle
        if hasattr(bundle, "predicates") and bundle.predicates is not None and spec.kind == "predicate_off":
            pred = replace(bundle.predicates, attention=bundle.predicates.attention * 0) if is_dataclass(bundle.predicates) else bundle.predicates
            bundle = replace(bundle, predicates=pred) if is_dataclass(bundle) else bundle
        return bundle

    def rerun_from_flow(self, bundle: Any, spec: InterventionSpecV2) -> Any:
        bundle = replace(bundle) if is_dataclass(bundle) else bundle
        if hasattr(bundle, "flow_state") and bundle.flow_state is not None and "flow_off" in spec.kind:
            flow = replace(bundle.flow_state, semantic_probs=bundle.flow_state.semantic_probs * 0) if is_dataclass(bundle.flow_state) else bundle.flow_state
            bundle = replace(bundle, flow_state=flow) if is_dataclass(bundle) else bundle
        return bundle

    def rerun_from_memory(self, bundle: Any, spec: InterventionSpecV2) -> Any:
        bundle = replace(bundle) if is_dataclass(bundle) else bundle
        if hasattr(bundle, "reason_memory") and bundle.reason_memory is not None:
            memory = replace(bundle.reason_memory, values=bundle.reason_memory.values * (1.0 - spec.strength)) if is_dataclass(bundle.reason_memory) else bundle.reason_memory
            bundle = replace(bundle, reason_memory=memory) if is_dataclass(bundle) else bundle
        return bundle


def zero_traffic_factor(tensor: Tensor, factor_idx: int) -> Tensor:
    out = tensor.clone()
    out[..., factor_idx] = 0
    return out


def delta_ce(logits: Tensor, cf_logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(cf_logits, labels) - F.cross_entropy(logits, labels)
