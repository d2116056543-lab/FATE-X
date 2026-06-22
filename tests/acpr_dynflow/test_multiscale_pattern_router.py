from ._helper import make_output

def test_multiscale_outputs():
    _,_,out=make_output()
    assert set(out.covariates.multiscale) == {'scale1','scale2','scale4'}

