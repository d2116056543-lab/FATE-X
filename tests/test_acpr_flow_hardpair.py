import torch
import pytest

from fate_x.acpr_flow.temporal_hard_pair import TemporalHardPairQueue


def test_hardpair_queue_caps_and_budget():
    q = TemporalHardPairQueue(hidden_dim=8, queue_size=4, pair_budget_ratio=0.08)
    q.enqueue(torch.randn(6, 8), torch.randn(6, 8))
    assert q.reason_target_queue.shape[0] == 4
    out = q(torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8), base_loss=torch.tensor(10.0))
    assert out["hardpair_budgeted_loss"] <= 0.8 + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device queue regression requires GPU")
def test_hardpair_enqueue_preserves_queue_device_after_cuda_move():
    q = TemporalHardPairQueue(hidden_dim=8, queue_size=4, pair_budget_ratio=0.08).cuda()
    q.enqueue(torch.randn(2, 8, device="cuda"), torch.randn(2, 8, device="cuda"))

    assert q.reason_target_queue.device.type == "cuda"
    assert q.action_queue.device.type == "cuda"
