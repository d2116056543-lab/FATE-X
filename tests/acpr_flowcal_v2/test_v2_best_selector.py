from fate_x.engine.train_acpr_flowcal_v2 import TestBestSelector


def test_best_selector_prefers_explanation_then_control_safe_tuple():
    selector = TestBestSelector()
    assert selector.update({"CIDEr_exp": 0.1, "CIDEr_des": 0.2, "METEOR_exp": 0.1, "speed_rmse": 1.0, "course_rmse": 1.0})
    assert not selector.update({"CIDEr_exp": 0.05, "CIDEr_des": 1.0, "METEOR_exp": 1.0, "speed_rmse": 0.0, "course_rmse": 0.0})
