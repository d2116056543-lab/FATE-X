import torch

from fate_x.acpr_flow_v2.axis_aware_flow_composer import derive_axis_direction_targets


def test_axis_direction_targets_use_named_signal_order():
    targets = torch.zeros(2, 32, 2)
    targets[:, :, 1] = 1.0
    out = derive_axis_direction_targets(targets, ["course", "speed"])
    assert {"longitudinal", "lateral", "direction"} <= set(out)
    assert out["longitudinal"].shape[0] == 2
