import torch

from fate_x.acpr_flow.region_priors import build_factor_support, build_region_prior_grid


def test_region_priors_normalized_and_distinct():
    priors = build_region_prior_grid(9, 11)
    assert priors.shape == (32, 9, 11)
    assert torch.allclose(priors.sum((-1, -2)), torch.ones(32), atol=1e-5)
    assert not torch.allclose(priors[0], priors[15])


def test_factor_support_has_support_and_contradiction():
    support, contradiction = build_factor_support()
    assert support.shape == (13, 32)
    assert contradiction.shape == (13, 32)
    assert support.sum() > 0
    assert contradiction.sum() > 0
