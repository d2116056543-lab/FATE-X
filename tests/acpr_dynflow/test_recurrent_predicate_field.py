from ._helper import make_output

def test_recurrent_predicate_field_temporal():
    _,_,out=make_output()
    assert out.predicates.query_states.shape[1] == 32

