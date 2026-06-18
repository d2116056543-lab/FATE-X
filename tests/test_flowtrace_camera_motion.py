import torch
from fate_x.models.robust_motion import weighted_geometric_median


def test_weighted_geometric_median_recovers_common_motion():
    pts = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.2, 0.2], [5.0, 5.0]]])
    weights = torch.tensor([[1.0, 1.0, 1.0, 0.01]])
    med = weighted_geometric_median(pts, weights, iterations=3)
    assert torch.allclose(med[0], torch.tensor([1.0, 0.0]), atol=0.25)
