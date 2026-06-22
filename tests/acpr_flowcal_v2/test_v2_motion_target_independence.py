import torch

from fate_x.acpr_flow_v2.adapt_motion_backbone import ADAPTMotionBackbone


def test_motion_predict_does_not_accept_or_depend_on_targets():
    model = ADAPTMotionBackbone(input_dim=8, hidden_dim=16)
    feats = torch.randn(1, 32, 8)
    pred1, _ = model.predict(feats, steps=32)
    pred2, _ = model.predict(feats, steps=32)
    assert torch.allclose(pred1, pred2)
