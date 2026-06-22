from ._helper import make_output

def test_text_losses_separate():
    _,_,out=make_output()
    assert 'action_text' in out.loss_components and 'explanation_text' in out.loss_components

