import torch


def test_temporal_shuffle_changes_sequence_order():
    x = torch.arange(8)
    assert not torch.equal(x, x.flip(0))
