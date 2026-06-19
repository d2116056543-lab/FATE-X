import torch

from fate_x.acpr_flow.reason_memory import ReasonMemory


def test_reason_memory_has_32_local_13_flow_1_null_tokens():
    mem = ReasonMemory(state_dim=16, hidden_dim=32)
    pred = torch.randn(2, 32, 16)
    flow = torch.randn(2, 13, 16)
    attn = torch.softmax(torch.randn(2, 13, 32), dim=-1)
    out = mem(pred, flow, attn)
    assert out["local_reason_memory"].shape == (2, 32, 32)
    assert out["flow_reason_memory"].shape == (2, 13, 32)
    assert out["null_reason_memory"].shape == (2, 1, 32)
    assert out["reason_memory"].shape == (2, 46, 32)
    assert out["reason_memory_mask"].sum().item() == 92
