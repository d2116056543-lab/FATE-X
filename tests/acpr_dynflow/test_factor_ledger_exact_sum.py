import torch
from ._helper import make_output

def test_factor_ledger_exact_sum():
    _,_,out=make_output()
    assert torch.allclose(out.ledger.final_prediction_normalized, out.ledger.global_prediction_normalized + out.ledger.factor_contributions_normalized.sum(2), atol=1e-5)
    assert torch.allclose(out.ledger.final_prediction_raw, out.ledger.global_prediction_raw + out.ledger.factor_contributions_raw.sum(2), atol=1e-5)

