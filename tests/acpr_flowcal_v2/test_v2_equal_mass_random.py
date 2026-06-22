import torch


def test_equal_mass_random_baseline_has_fixed_seed_reproducibility():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert torch.equal(torch.randperm(10, generator=g1), torch.randperm(10, generator=g2))
