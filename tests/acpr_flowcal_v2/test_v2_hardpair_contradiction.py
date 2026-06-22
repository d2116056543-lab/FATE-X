import torch

from fate_x.acpr_flow_v2.temporal_hardpair import ContradictionAwareTemporalHardPair


def test_hardpair_queue_does_not_update_in_eval_mode():
    hp = ContradictionAwareTemporalHardPair(queue_size=4)
    hp.eval()
    hp.enqueue(torch.randn(2, 3))
    assert hp.queue.numel() == 0
