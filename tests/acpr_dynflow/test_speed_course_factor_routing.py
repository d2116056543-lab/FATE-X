from ._helper import make_output

def test_separate_attentions():
    _,_,out=make_output()
    assert out.ledger.speed_factor_attention.shape[-1] == 13
    assert out.ledger.course_factor_attention.shape[-1] == 13

