from ._helper import make_output

def test_evidence_normalizes():
    _,_,out=make_output()
    s=out.predicates.evidence_maps.flatten(-2).sum(-1)
    assert float((s-1).abs().max()) < 1e-4

