import torch
from fate_x.models.sinkhorn_transport import LogSinkhornTransport


def test_sinkhorn_shapes_and_grad():
    m = LogSinkhornTransport(8, matching_dim=4)
    a = torch.randn(2, 9, 8, requires_grad=True)
    b = a.detach().clone() + 0.01 * torch.randn(2, 9, 8)
    out = m(a, b)
    assert out["transport"].shape == (2, 10, 10)
    assert torch.isfinite(out["transport"]).all()
    out["matched_transport"].sum().backward()
    assert m.proj.weight.grad is not None
