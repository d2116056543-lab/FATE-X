import torch

from fate_x.acpr_flow_v2.prefix_future import build_prefix_bundle_from_precomputed_grids


def test_prefix_future_uses_precomputed_grids_without_backbone_call():
    bundle = build_prefix_bundle_from_precomputed_grids(torch.randn(1, 32, 2, 2, 4), prefix_frames=24, target_frames=8)
    assert bundle["prefix"].shape[1] == 24
    assert bundle["future"].shape[1] == 8
