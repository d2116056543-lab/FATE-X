from ._helper import make_output

def test_covariates_have_required_groups():
    _,_,out=make_output()
    assert out.covariates.raw_covariates.shape[-1] == 10

