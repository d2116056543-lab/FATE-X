import torch
from fate_x.models.token_pmt_adapter import TokenPMTAdapter


def test_pmt_gate_zero_and_zero_init():
    pmt = TokenPMTAdapter(hidden_dim=16, state_dim=8, rank=4)
    h = torch.randn(2, 6, 16)
    z = torch.randn(2, 3, 8)
    r = torch.randn(2, 8)
    tt = torch.tensor([[0,0,1,1,1,1],[0,1,0,1,0,1]])
    y, _ = pmt(h, z, r, tt, scale=0.0)
    assert torch.equal(y, h)
    assert torch.allclose(pmt.out_proj.weight, torch.zeros_like(pmt.out_proj.weight))
