import torch

from fate_x.acpr_flow_v2.local_partial_transport import LocalPartialTransportV2
from fate_x.acpr_flow_v2.temporal_predicate_tracker import TransportedNamedPredicateTracker


def test_transported_predicate_tracker_outputs_32_named_trajectories():
    grid = torch.randn(1, 3, 3, 3, 4)
    transport = LocalPartialTransportV2(dim=4, local_radius=1)(grid)
    out = TransportedNamedPredicateTracker(dim=4, num_predicates=32)(grid, transport)
    assert len(out.names) == 32
    assert out.attention.shape[:3] == (1, 3, 32)
    assert out.descriptor.shape[0] == 1
