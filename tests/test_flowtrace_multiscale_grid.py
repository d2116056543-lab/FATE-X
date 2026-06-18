import torch
from fate_x.models.multiscale_video_grid import MultiScaleVideoGrid


def test_multiscale_grid_fuses_to_fine_resolution():
    m = MultiScaleVideoGrid(16, 32, 8)
    fine = torch.randn(2, 16, 4, 6, 6)
    coarse = torch.randn(2, 32, 4, 3, 3)
    out = m(fine, coarse)
    assert out["fine_grid"].shape == (2, 4, 6, 6, 8)
    assert out["coarse_grid"].shape == (2, 4, 3, 3, 8)
    assert out["fused_grid"].shape == out["fine_grid"].shape
