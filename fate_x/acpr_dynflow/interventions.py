from __future__ import annotations

from typing import Any


def run_intervention(model: Any, batch: Any, intervention: str) -> Any:
    if intervention not in {
        "all_flow_off",
        "regime_off",
        "phase_off",
        "source_off",
        "top_factor_off",
        "predicate_off",
        "evidence_tube_off",
        "random_equal_mass",
        "temporal_shuffle",
        "temporal_reverse",
        "lag_disabled",
        "last_frame_only",
    }:
        raise ValueError(f"unsupported intervention {intervention}")
    return model(batch, intervention=intervention)

