import torch

from fate_x.acpr_flow.temporal_hard_pair import TemporalHardPairQueue


def test_pair_loss_not_double_added_and_respects_budget_zero_queue():
    q = TemporalHardPairQueue(hidden_dim=8, pair_budget_ratio=0.08)
    out = q(torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8), base_loss=torch.tensor(1.0))
    assert out["hardpair_raw_loss"].item() == 0.0
    assert out["hardpair_budgeted_loss"].item() == 0.0
