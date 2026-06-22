import torch

from fate_x.acpr_flow_v2.adapt_motion_backbone import ADAPTMotionBackbone


def test_adapt_motion_predict_is_deterministic_for_same_features():
    model = ADAPTMotionBackbone(input_dim=8, hidden_dim=16)
    feats = torch.randn(2, 32, 8)
    pred1, hid1 = model.predict(feats, steps=32)
    pred2, hid2 = model.predict(feats, steps=32)
    assert torch.allclose(pred1, pred2)
    assert torch.allclose(hid1, hid2)
    assert pred1.shape == (2, 32, 2)
