from __future__ import annotations

import torch

from fate_x.engine.acpr_control_eval import compute_control_metrics
from fate_x.losses.acpr_flowcal_losses import masked_l2_loss


def test_compute_control_metrics_reports_adapt_continuous_metrics_without_decision_proxy() -> None:
    target = torch.tensor(
        [
            [[0.0, 10.0], [0.1, 9.0], [0.2, 8.0]],
            [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]],
            [[0.0, 2.0], [0.3, 3.0], [0.8, 4.0]],
        ],
        dtype=torch.float32,
    )
    pred = target.clone()
    pred[0, :, 1] -= 0.1
    pred[2, -1, 0] = 0.7

    metrics = compute_control_metrics(
        pred,
        target,
        signal_names=["course", "speed"],
        invalid_value=-1.0,
    )

    assert metrics["sample_count"] == 3
    assert metrics["valid_value_count"] == 18
    assert metrics["metric_family"] == "adapt_continuous_control"
    assert metrics["signals"]["speed"]["rmse"] > 0.0
    assert metrics["signals"]["speed"]["acc_at_0.5"] == 1.0
    assert "speed_decision" not in metrics
    assert "course_decision" not in metrics
    assert metrics["diagnostic_decision_proxy_available"] is False


def test_compute_control_metrics_ignores_invalid_values() -> None:
    target = torch.tensor([[[0.0, 1.0], [-1.0, -1.0]]], dtype=torch.float32)
    pred = torch.tensor([[[0.0, 1.2], [100.0, 100.0]]], dtype=torch.float32)

    metrics = compute_control_metrics(pred, target, signal_names=["course", "speed"])

    assert metrics["valid_value_count"] == 2
    assert metrics["signals"]["course"]["rmse"] == 0.0
    assert metrics["signals"]["speed"]["rmse"] > 0.0


def test_compute_control_metrics_and_loss_use_circular_course_error() -> None:
    target = torch.tensor([[[350.0, 1.0]]], dtype=torch.float32)
    pred = torch.tensor([[[10.0, 1.0]]], dtype=torch.float32)

    metrics = compute_control_metrics(pred, target, signal_names=["course", "speed"])

    assert metrics["signals"]["course"]["rmse"] == 20.0
    assert metrics["signals"]["course"]["mae"] == 20.0
    assert metrics["signals"]["course"]["acc_at_10"] == 0.0
    assert metrics["signals"]["course"]["acc_at_5"] == 0.0
    assert metrics["signals"]["speed"]["rmse"] == 0.0

    loss = masked_l2_loss(pred, target, signal_names=["course", "speed"])

    assert float(loss) == 200.0
