from fate_x.engine.eval_acpr_flowcal_v2 import compute_control_metrics
import torch


def test_compute_control_metrics_reports_adapt_style_thresholds_and_no_fake_cider():
    pred = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    target = torch.tensor([[[0.0, 1.5], [1.0, 5.0]]])

    metrics = compute_control_metrics(pred, target)

    assert metrics["course_rmse"] > 0
    assert metrics["speed_rmse"] > 0
    assert metrics["course_mae"] == 0.5
    assert metrics["speed_mae"] == 1.25
    for key in [
        "course_acc@0.1",
        "course_acc@0.5",
        "course_acc@1",
        "course_acc@5",
        "course_acc@10",
        "speed_acc@0.1",
        "speed_acc@0.5",
        "speed_acc@1",
        "speed_acc@5",
        "speed_acc@10",
    ]:
        assert key in metrics
    assert "CIDEr_des" not in metrics
