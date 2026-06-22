from ._helper import make_output

def test_global_stream_shape():
    _,_,out=make_output()
    assert out.ledger.global_prediction_normalized.shape == (1,32,2)

