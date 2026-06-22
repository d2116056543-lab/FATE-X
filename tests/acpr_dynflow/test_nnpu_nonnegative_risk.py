import torch
from fate_x.acpr_dynflow.nnpu_calalign import PULabels, nnpu_risk

def test_nnpu_nonnegative():
    logits=torch.randn(2,32)
    labels=PULabels(torch.zeros(2,32), torch.zeros(2,32), torch.ones(2,32))
    assert nnpu_risk(logits, labels).item() >= 0

