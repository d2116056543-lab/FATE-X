import torch

from fate_x.acpr_flow.local_partial_transport import LocalPartialTransport


def test_local_partial_transport_uses_25_neighbors_plus_dustbin_and_gradients():
    module = LocalPartialTransport(in_dim=8, matching_dim=4, local_radius=2)
    x = torch.randn(2, 4, 5, 5, 8, requires_grad=True)
    out = module(x)
    probs = out["local_transport_probs"]
    assert probs.shape == (2, 3, 25, 26)
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=1e-5)
    assert module.last_memory_report["dense_global_matrix"] is False
    loss = probs[..., :-1].sum()
    loss.backward()
    assert module.proj.weight.grad.abs().sum() > 0
    assert out["dustbin_prob"].shape == (2, 3, 25)
