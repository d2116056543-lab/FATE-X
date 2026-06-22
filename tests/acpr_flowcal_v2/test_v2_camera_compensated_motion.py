import torch

from fate_x.acpr_flow_v2.local_partial_transport import LocalPartialTransportV2


def test_transport_exposes_common_shift_for_camera_compensation():
    out = LocalPartialTransportV2(dim=4, local_radius=1)(torch.randn(2, 3, 4, 4, 4))
    assert out.common_shift.shape == (2, 2, 2)
    assert torch.isfinite(out.expected_displacement).all()
