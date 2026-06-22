from ._helper import make_output

def test_contribution_alignment_loss_nonnegative():
    _,_,out=make_output()
    assert out.loss_components['contribution_alignment'].item() >= 0

