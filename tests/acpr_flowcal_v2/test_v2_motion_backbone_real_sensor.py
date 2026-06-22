from fate_x.acpr_flow_v2.adapt_motion_backbone import ADAPTMotionBackbone
from src.modeling.load_sensor_pred_head import Sensor_Pred_Head
import torch


def test_v2_motion_backbone_wraps_released_adapt_sensor_head():
    model = ADAPTMotionBackbone(input_dim=32, hidden_dim=32, output_dim=2)

    assert isinstance(model.sensor_head, Sensor_Pred_Head)
    dense = torch.randn(2, 32, 32)
    pred, hidden = model.predict(dense, steps=32)
    assert pred.shape == (2, 32, 2)
    assert hidden.shape[:2] == (2, 32)
