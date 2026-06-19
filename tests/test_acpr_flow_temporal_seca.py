import torch

from fate_x.acpr_flow.temporal_seca import TemporalSECA


def test_temporal_seca_zero_gate_preserves_hidden_but_has_gate_gradients():
    seca = TemporalSECA(hidden_dim=32)
    hidden = torch.randn(2, 10, 32, requires_grad=True)
    memory = torch.randn(2, 46, 32, requires_grad=True)
    out, info = seca(hidden, memory, text_len=6)
    assert torch.allclose(out, hidden, atol=1e-6)
    assert float(info["image_hidden_max_diff"]) == 0.0
    out.sum().backward()
    assert seca.gamma_action_raw.grad.abs().sum() > 0
    assert seca.out.weight.abs().sum() > 0
