import torch

from fate_x.acpr_flow_v2.semantic_gradient_firewall import scaled_gradient


def test_firewall_preserves_forward_tensor():
    x = torch.randn(4, requires_grad=True)
    assert torch.allclose(scaled_gradient(x, 0.0), x)
