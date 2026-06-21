from __future__ import annotations

import torch
import pytest

from fate_x.engine.train_acpr_flowcal_pp import summarize_traffic_flow_audit


def test_traffic_flow_audit_reports_prediction_delta_correlations_and_diagnostics() -> None:
    flow_probs = torch.tensor(
        [
            [0.95, 0.05],
            [0.80, 0.20],
            [0.20, 0.80],
            [0.05, 0.95],
        ],
        dtype=torch.float32,
    )
    predicate_probs = torch.tensor(
        [
            [0.10],
            [0.30],
            [0.70],
            [0.90],
        ],
        dtype=torch.float32,
    )
    pred_control = torch.tensor(
        [
            [[0.0, 1.0], [0.1, 1.4], [0.2, 2.0]],
            [[0.0, 1.0], [0.1, 1.3], [0.2, 1.8]],
            [[0.0, 3.0], [0.2, 2.8], [0.4, 2.4]],
            [[0.0, 3.0], [0.3, 2.7], [0.6, 2.0]],
        ],
        dtype=torch.float32,
    )
    target_control = pred_control.clone()

    audit = summarize_traffic_flow_audit(
        flow_probs,
        predicate_probs,
        pred_control,
        target_control,
        flow_factor_names=["clear_open_flow", "queue_congestion"],
        predicate_names=["road_crowded"],
        signal_names=["course", "speed"],
    )

    clear = audit["flow_factors"]["clear_open_flow"]
    queue = audit["flow_factors"]["queue_congestion"]
    assert clear["pred_speed_delta_corr"] is not None
    assert queue["pred_speed_delta_corr"] is not None
    assert clear["pred_speed_delta_corr"] > 0.9
    assert queue["pred_speed_delta_corr"] < -0.9
    assert clear["pred_speed_delta_corr_reason"] == "ok"
    assert queue["pred_speed_delta_corr_reason"] == "ok"

    delta = audit["delta_stats"]["speed"]
    assert delta["valid_pred_delta_count"] == 4
    assert delta["pred_delta_std"] > 0.0
    assert delta["target_delta_std"] > 0.0


def test_traffic_flow_audit_uses_circular_course_delta() -> None:
    flow_probs = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    control = torch.tensor(
        [
            [[350.0, 1.0], [10.0, 2.0]],
            [[10.0, 1.0], [350.0, 2.0]],
        ],
        dtype=torch.float32,
    )

    audit = summarize_traffic_flow_audit(
        flow_probs,
        None,
        control,
        control,
        flow_factor_names=["flow_factor"],
        signal_names=["course", "speed"],
    )

    course_delta = audit["delta_stats"]["course"]
    assert course_delta["pred_delta_min"] == pytest.approx(-20.0)
    assert course_delta["pred_delta_max"] == pytest.approx(20.0)
    assert course_delta["pred_delta_std"] == pytest.approx(20.0)
