from ._helper import make_output

def test_named_output_shapes():
    _,_,out=make_output()
    assert out.predicates.logits.shape[-1] == 32
    assert out.flow.factor_tokens.shape[2] == 13
    assert out.ledger.final_prediction_normalized.shape[-1] == 2

