import torch

from fate_x.acpr_flow_v2.axis_aware_flow_composer import derive_axis_direction_targets


def test_lateral_direction_sign_distinguishes_left_and_right():
    targets = torch.zeros(3, 32, 2)
    targets[0, :, 0] = -1
    targets[1, :, 0] = 0
    targets[2, :, 0] = 1
    out = derive_axis_direction_targets(targets, ["course", "speed"])
    assert out["direction"].argmax(-1).tolist() == [0, 1, 2]
