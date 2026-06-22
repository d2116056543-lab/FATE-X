from ._helper import make_output

def test_response_lag_weights():
    _,_,out=make_output()
    assert out.flow.response_lag_weights.shape[-1] == 4

