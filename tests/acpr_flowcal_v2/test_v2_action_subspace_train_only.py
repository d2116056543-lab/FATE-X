import torch

from fate_x.acpr_flow_v2.contextual_reason_target import ActionSubspaceTracker


def test_action_subspace_tracker_state_roundtrip():
    tracker = ActionSubspaceTracker(rank=2)
    tracker.update(torch.randn(4, 8))
    tracker.finalize_epoch()
    state = tracker.state_dict()
    clone = ActionSubspaceTracker(rank=2)
    clone.load_state_dict(state)
    assert clone.state_dict().keys() == state.keys()
