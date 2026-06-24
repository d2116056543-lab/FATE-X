from __future__ import annotations


def traffic_flow_utilization(flowoff_rmse: float, full_rmse: float) -> float:
    return float(flowoff_rmse) - float(full_rmse)


def evidence_specificity(evidence_delta: float, random_deltas: list[float]) -> float:
    baseline = sum(random_deltas) / max(1, len(random_deltas))
    return float(evidence_delta) - baseline
