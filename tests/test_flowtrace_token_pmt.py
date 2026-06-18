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


def test_pmt_aligns_full_sequence_token_type_ids_to_hidden_length():
    pmt = TokenPMTAdapter(hidden_dim=16, state_dim=8, rank=4)
    h = torch.randn(2, 30, 16)
    z = torch.randn(2, 3, 8)
    r = torch.randn(2, 8)
    tt = torch.zeros(2, 814, dtype=torch.long)
    tt[:, 15:30] = 1

    y, info = pmt(h, z, r, tt, scale=1.0)

    assert y.shape == h.shape
    assert info["pmt_gate"].shape == h.shape[:2]
    assert torch.all(info["pmt_gate"][:, :15] == pmt.action_gate_max)
    assert torch.all(info["pmt_gate"][:, 15:] == pmt.explanation_gate_max)


def test_pmt_pads_text_only_token_type_ids_to_full_hidden_length():
    pmt = TokenPMTAdapter(hidden_dim=16, state_dim=8, rank=4)
    h = torch.randn(2, 814, 16)
    z = torch.randn(2, 3, 8)
    r = torch.randn(2, 8)
    tt = torch.zeros(2, 30, dtype=torch.long)
    tt[:, 15:30] = 1

    y, info = pmt(h, z, r, tt, scale=1.0)

    assert y.shape == h.shape
    assert info["pmt_gate"].shape == h.shape[:2]
    assert torch.all(info["pmt_gate"][:, :15] == pmt.action_gate_max)
    assert torch.all(info["pmt_gate"][:, 15:30] == pmt.explanation_gate_max)
    assert torch.all(info["pmt_gate"][:, 30:] == pmt.action_gate_max)
