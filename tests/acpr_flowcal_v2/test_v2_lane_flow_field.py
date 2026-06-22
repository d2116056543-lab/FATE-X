import torch

from fate_x.acpr_flow_v2.lane_flow_field import PredicateConditionedLaneFlowField
from fate_x.acpr_flow_v2.local_partial_transport import LocalPartialTransportV2
from fate_x.acpr_flow_v2.temporal_predicate_tracker import TransportedNamedPredicateTracker


def test_lane_flow_field_has_three_ordered_regions():
    grid = torch.randn(1, 3, 3, 3, 4)
    pred = TransportedNamedPredicateTracker(dim=4, num_predicates=32)(grid, LocalPartialTransportV2(dim=4, local_radius=1)(grid))
    out = PredicateConditionedLaneFlowField(dim=4)(pred, grid)
    assert out.region_names == ("left", "center", "right")
    assert out.occupancy.shape[:3] == (1, 3, 3)
