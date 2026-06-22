from ._helper import make_output

def test_e2e_direct_image_smoke():
    _,batch,out=make_output()
    assert list(batch.frames.shape) == [1,32,3,224,224]
    assert out.total_loss.isfinite()

