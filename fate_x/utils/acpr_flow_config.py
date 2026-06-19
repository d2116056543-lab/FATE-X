from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


FORBIDDEN_LEGACY_TRUE = [
    "legacy_flowtrace_v1_enabled",
    "legacy_token_pmt_enabled",
    "legacy_sinkhorn_transport_enabled",
    "learn_mask_enabled",
]


def load_acpr_flow_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_acpr_flow_config(cfg)
    return cfg


def validate_acpr_flow_config(cfg: dict[str, Any]) -> None:
    data = cfg.get("data", {})
    if data.get("direct_image_training") is not True:
        raise ValueError("ACPR formal path requires direct_image_training=true")
    for key in ("feature_cache_enabled", "token_cache_enabled", "build_cache_before_training"):
        if data.get(key) is not False:
            raise ValueError(f"ACPR formal path requires {key}=false")
    legacy = cfg.get("legacy", {})
    for key in FORBIDDEN_LEGACY_TRUE:
        if legacy.get(key) is not False:
            raise ValueError(f"legacy.{key} must be false")
    if cfg.get("evaluation", {}).get("eval_splits") != ["test"]:
        raise ValueError("ACPR formal protocol is test-selected and must not create val loader")
    if cfg.get("supervisor", {}).get("foreground_only") is not True:
        raise ValueError("foreground supervisor is required")
