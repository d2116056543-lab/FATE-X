import torch

from fate_x.acpr_flow.temporal_predicate_field import TemporalPredicateEmbeddingField


def test_descriptor_distinguishes_temporal_patterns():
    field = TemporalPredicateEmbeddingField(state_dim=8)
    inc = torch.arange(5.0).view(1, 5, 1, 1).expand(1, 5, 32, 8)
    dec = inc.flip(1)
    rel = torch.zeros(1, 4, 32, 2)
    conf = torch.ones(1, 5, 32)
    d_inc = field._descriptor(inc, rel, conf)["descriptor"]
    d_dec = field._descriptor(dec, rel, conf)["descriptor"]
    assert not torch.allclose(d_inc, d_dec)
