import torch

from fate_x.acpr_flow_v2.semantic_gradient_firewall import scaled_gradient


def test_firewall_gradient_ratio_matches_requested_scale():
    x = torch.ones(1, requires_grad=True)
    scaled_gradient(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([0.5]))
