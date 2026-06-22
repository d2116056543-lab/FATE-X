from fate_x.engine.train_acpr_flowcal_v2 import TestBestSelector
from fate_x.engine.train_acpr_flowcal_v2 import evaluate_after_epoch
from fate_x.acpr_flow_v2.config import FlowCalV2Config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.engine.acpr_flowcal_v2_data import build_v2_dataloader


def test_best_selector_uses_test_metric_tuple():
    s = TestBestSelector()
    assert s.update({"CIDEr_exp": 0.1, "CIDEr_des": 0.2, "METEOR_exp": 0.1, "speed_rmse": 1.0, "course_rmse": 1.0})
    assert s.update({"CIDEr_exp": 0.2, "CIDEr_des": 0.1, "METEOR_exp": 0.1, "speed_rmse": 1.0, "course_rmse": 1.0})


def test_epoch_eval_does_not_infer_text_metrics_from_loss():
    cfg = FlowCalV2Config(use_real_video_swin=False, hidden_dim=16, state_dim=16, text_hidden_dim=32, text_vocab_size=101, num_frames=4)
    model = ACPRFlowCalV2Model(cfg)
    loader = build_v2_dataloader("test", batch_size=1, synthetic=True, length=1, vocab=101)
    metrics = evaluate_after_epoch(model, loader, epoch=0, device="cpu")
    assert "CIDEr_des" not in metrics
    assert "CIDEr_exp" not in metrics
    assert metrics["text_metrics_available"] is False
    assert "speed_rmse" in metrics
    assert "course_rmse" in metrics
