import torch

from fate_x.acpr_flow_v2.local_partial_transport import LocalPartialTransportV2, warp_source_map_to_current


def test_transport_warp_preserves_shape_and_finiteness():
    transport = LocalPartialTransportV2(dim=4, local_radius=1)(torch.randn(1, 2, 3, 3, 4))
    warped = warp_source_map_to_current(torch.randn(1, 2, 3, 3), transport, step=0)
    assert warped.shape == (1, 2, 3, 3)
    assert torch.isfinite(warped).all()
