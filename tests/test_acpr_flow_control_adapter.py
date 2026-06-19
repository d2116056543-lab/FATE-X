import torch

from fate_x.acpr_flow.reason_control_adapter import ReasonControlAdapter


def test_control_adapter_zero_gate_matches_base_and_has_gate_gradient():
    adapter = ReasonControlAdapter(hidden_dim=32, signals=2)
    base = torch.randn(2, 32, 2)
    hidden = torch.randn(2, 32, 32)
    memory = torch.randn(2, 46, 32)
    out = adapter(base, hidden, memory)
    assert torch.allclose(out["control_final_prediction"], base, atol=1e-6)
    out["control_final_prediction"].sum().backward()
    assert adapter.gate_raw.grad.abs().sum() > 0
