from ._helper import make_output

def test_contribution_alignment_loss_nonnegative():
    _,_,out=make_output()
    assert out.loss_components['contribution_alignment'].item() >= 0

def test_contribution_alignment_supports_real_batch_size():
    _, batch, out = make_output(length=10)
    assert batch.frames.shape[0] == 10
    assert out.text.action_to_factor_attention.shape[0] == 10
    assert out.ledger.speed_factor_attention.shape[1] == 32
    assert out.ledger.speed_factor_attention.shape[-1] == 13
    assert out.loss_components['contribution_alignment'].isfinite()

