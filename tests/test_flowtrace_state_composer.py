import torch
from fate_x.models.dynamic_traffic_state_composer import DynamicTrafficStateComposer


def test_state_composer_maps_are_weighted_tracks():
    m = DynamicTrafficStateComposer(state_dim=12, num_states=3, num_tracks=4, heads=2)
    tokens = torch.randn(2, 5, 4, 12, requires_grad=True)
    attn = torch.softmax(torch.randn(2, 5, 4, 3, 3), dim=-1)
    rel = torch.randn(2, 4, 4, 2)
    unmatched = torch.rand(2, 5, 4)
    out = m(tokens, attn, rel, unmatched)
    assert out["state_memory"].shape == (2, 3, 12)
    recon = torch.einsum("btkl,btlhw->btkhw", out["state_track_weights"], attn)
    assert torch.allclose(out["state_evidence_maps"], recon, atol=1e-5)
    out["state_memory"].sum().backward()
    assert m.state_queries.grad is not None
