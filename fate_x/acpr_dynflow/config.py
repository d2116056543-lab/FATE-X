from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


ALLOWED_TOP_LEVEL = {
    "config_version",
    "experiment_name",
    "protocol_tag",
    "repository",
    "paths",
    "data",
    "model",
    "loss",
    "optimization",
    "memory_probe",
    "evaluation",
    "faithfulness",
    "visualization",
    "preflight",
    "supervisor",
}


@dataclass
class DynFlowConfig:
    raw: dict[str, Any]
    path: str
    consumer_manifest: dict[str, str]

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


def _flatten(prefix: str, obj: Any, out: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    else:
        out[prefix] = "runtime_or_audit_consumer"


def load_dynflow_config(path: str | Path) -> DynFlowConfig:
    path = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required for ACPR-DynFlow config loading")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - ALLOWED_TOP_LEVEL)
    if unknown:
        raise KeyError(f"Unknown ACPR-DynFlow config top-level keys: {unknown}")
    manifest: dict[str, str] = {}
    _flatten("", raw, manifest)
    return DynFlowConfig(raw=raw, path=str(path), consumer_manifest=manifest)

