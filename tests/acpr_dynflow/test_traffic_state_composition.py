from ._helper import make_output

def test_traffic_state_13():
    _,_,out=make_output()
    assert len(out.flow.factor_names) == 13
    assert out.flow.factor_to_predicate.shape[-1] == 32

