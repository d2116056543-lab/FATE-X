import torch

from fate_x.acpr_flow_v2.local_partial_transport import LocalPartialTransportV2
from fate_x.acpr_flow_v2.temporal_predicate_tracker import TransportedNamedPredicateTracker


def test_dynamic_descriptor_parts_include_temporal_terms():
    grid = torch.randn(1, 4, 3, 3, 4)
    out = TransportedNamedPredicateTracker(dim=4, num_predicates=32)(grid, LocalPartialTransportV2(dim=4, local_radius=1)(grid))
    assert {"trend", "volatility", "presence_rate"} & set(out.descriptor_parts)
