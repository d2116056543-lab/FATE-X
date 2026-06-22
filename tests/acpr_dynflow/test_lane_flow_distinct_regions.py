from ._helper import make_output

def test_lane_flow_not_repeated():
    _,_,out=make_output()
    lane=out.predicates.lane_mass
    assert lane.shape[-1] == 3

