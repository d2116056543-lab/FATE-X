import torch
from fate_x.acpr_dynflow.ego_motion import estimate_common_shift

def test_known_translation_shape():
    x=torch.randn(1,4,3,3,2)
    assert estimate_common_shift(x).shape == (1,4,2)

