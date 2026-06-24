from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _walk(prefix: str, value: Any, out: dict[str, dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            _walk(dotted, child, out)
    else:
        out[prefix] = {"consumer": _consumer_for(prefix)}


def _consumer_for(dotted: str) -> str:
    routing = {
        "model.semantic_consolidation": "SemanticTokenConsolidator",
        "model.traffic": "PatternLagTrafficReasoner",
        "model.ledger": "ExactDecisionLedgerHead",
        "optimization.learning_rates": "build_optimizer_groups",
        "memory_throughput_probe": "probe_acpr_dynflow_swin_throughput",
        "evaluation": "eval_acpr_dynflow_swin",
        "paths": "acpr_dynflow_swin_data",
        "data": "DynFlowSwinDataModule",
        "loss": "acpr_dynflow_swin_losses",
        "supervisor": "supervise_acpr_dynflow_swin_foreground",
    }
    for prefix, consumer in routing.items():
        if dotted.startswith(prefix):
            return consumer
    return "config_resolved_manifest"


def build_config_consumer_manifest(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    _walk("", cfg, manifest)
    return manifest
