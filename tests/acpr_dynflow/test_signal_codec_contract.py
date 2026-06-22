import torch
from fate_x.acpr_dynflow.signal_codec import BDDSignalCodec

def test_signal_codec_roundtrip():
    x=torch.randn(2,32,2)
    c=BDDSignalCodec().fit(x)
    assert torch.allclose(c.decode(c.encode(x)), x, atol=1e-5)

