from types import SimpleNamespace

import torch

from src.modeling.load_sensor_pred_head import Sensor_Pred_Head


def _args():
    return SimpleNamespace(
        img_feature_dim=8,
        grid_feat=True,
        config_name="",
        model_name_or_path="models/bert-base-uncased",
        signal_types=["course", "speed"],
    )


def test_sensor_pred_head_exposes_target_independent_encode_predict():
    model = Sensor_Pred_Head(_args())
    feats = torch.randn(2, 5, 8)

    hidden = model.encode(feats)
    assert hidden.shape[:2] == (2, 5)
    assert hidden.shape[-1] == model.config.hidden_size

    pred = model.predict(feats, frame_num=3)
    assert pred.shape == (2, 3, 2)


def test_sensor_pred_head_forward_remains_backward_compatible():
    model = Sensor_Pred_Head(_args())
    feats = torch.randn(2, 5, 8)
    target = torch.randn(2, 2, 5)

    loss, pred, hidden = model(img_feats=feats, car_info=target, return_hidden=True)
    assert loss.ndim == 0
    assert pred.shape == (2, 5, 2)
    assert hidden.shape[:2] == (2, 5)
