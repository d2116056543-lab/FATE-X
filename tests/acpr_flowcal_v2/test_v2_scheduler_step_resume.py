import torch

from fate_x.engine.train_acpr_flowcal_v2 import StageAwareScheduler


def test_scheduler_state_roundtrip():
    p = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([p], lr=1e-3)
    sched = StageAwareScheduler(opt, total_steps=4)
    sched.step()
    state = sched.state_dict()
    clone = StageAwareScheduler(opt, total_steps=4)
    clone.load_state_dict(state)
    assert clone.step_count == sched.step_count
