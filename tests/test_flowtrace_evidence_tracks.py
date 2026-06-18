import torch
from fate_x.models.transported_evidence_tracks import TransportedEvidenceTracks


def test_evidence_tracks_normalize_and_grad():
    m = TransportedEvidenceTracks(16, state_dim=12, num_tracks=3)
    x = torch.randn(2, 4, 5, 5, 16, requires_grad=True)
    out = m(x)
    assert out["track_attention"].shape == (2, 4, 3, 5, 5)
    mass = out["track_attention"].flatten(-2).sum(-1)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-4)
    out["track_tokens"].sum().backward()
    assert m.track_queries.grad is not None
