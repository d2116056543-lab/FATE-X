import torch

from fate_x.acpr_flow_v2.semantic_gradient_firewall import scaled_gradient


def test_scaled_gradient_changes_backward_scale_not_forward_value():
    x = torch.ones(3, requires_grad=True)
    y = scaled_gradient(x, 0.25)
    assert torch.allclose(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 0.25))
