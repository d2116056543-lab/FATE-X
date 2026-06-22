import torch

from fate_x.acpr_flow_v2.local_partial_transport import LocalPartialTransportV2


def test_local_transport_rows_sum_to_one_and_has_no_global_matrix():
    out = LocalPartialTransportV2(dim=4, local_radius=1)(torch.randn(1, 3, 4, 4, 4))
    assert out.probs.shape[-1] == 10
    assert torch.allclose(out.probs.sum(-1), torch.ones_like(out.probs[..., 0]), atol=1e-5)
    assert "transport_steps" in out.diagnostics
