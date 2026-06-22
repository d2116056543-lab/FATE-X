import torch

from fate_x.losses.acpr_flowcal_v2_losses import normalized_control_huber


def test_normalized_control_huber_is_finite():
    loss = normalized_control_huber(torch.zeros(2, 32, 2), torch.ones(2, 32, 2), {"std": torch.ones(2)})
    assert torch.isfinite(loss)
