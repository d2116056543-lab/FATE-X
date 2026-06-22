import torch

from fate_x.losses.acpr_flowcal_v2_losses import shortest_circular_delta


def test_course_circular_residual_wraps_at_pi():
    a = torch.tensor([3.13])
    b = torch.tensor([-3.13])
    assert shortest_circular_delta(a, b).abs().item() < 0.1
