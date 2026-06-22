from ._helper import make_output

def test_no_generated_gt_by_default():
    _,_,out=make_output()
    assert out.text.generated_action is None

