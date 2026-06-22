from ._helper import make_output

def test_explanation_attention_exists():
    _,_,out=make_output()
    assert out.text.explanation_to_factor_attention.shape[-1] == 13

