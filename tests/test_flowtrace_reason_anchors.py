import torch
from fate_x.models.reason_state_anchors import ridge_residualize, spherical_kmeans, ReasonStateAnchorBank


def test_reason_anchors_nonempty_and_distribution():
    a = torch.randn(32, 12)
    r = a * 0.5 + torch.randn(32, 12)
    residual = ridge_residualize(a, r)
    centers, labels = spherical_kmeans(residual, k=4, iterations=5)
    assert centers.shape == (4, 12)
    assert labels.unique().numel() >= 2
    bank = ReasonStateAnchorBank(centers)
    dist = bank(torch.randn(3, 4, 12))
    assert dist.shape == (3, 4)
    assert torch.allclose(dist.sum(-1), torch.ones(3), atol=1e-5)
