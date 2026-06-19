import torch

from fate_x.acpr_flow.temporal_predicate_field import TemporalPredicateEmbeddingField


def test_predicate_field_outputs_full_temporal_contract():
    field = TemporalPredicateEmbeddingField(state_dim=16)
    grid = torch.randn(2, 5, 4, 4, 16, requires_grad=True)
    out = field(grid)
    assert out["attention"].shape == (2, 5, 32, 4, 4)
    assert out["tokens"].shape == (2, 5, 32, 16)
    assert out["presence_logits"].shape == (2, 5, 32)
    assert out["trajectory_confidence"].shape == (2, 5, 32)
    assert out["relative_motion"].shape == (2, 4, 32, 2)
    assert out["descriptor"].shape == (2, 32, 16)
    assert torch.allclose(out["attention"].sum((-1, -2)), torch.ones(2, 5, 32), atol=1e-5)
    out["presence_logits"].sum().backward()
    assert field.queries.grad.abs().sum() > 0
