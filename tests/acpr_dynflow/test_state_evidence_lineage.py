from ._helper import make_output

def test_lineage_complete():
    _,_,out=make_output()
    assert len(out.flow.lineage) == 13

