import torch
from fate_x.models.flowtrace_pmt_model import FlowTracePMTModel
from fate_x.explain.flowtrace_intervention import FlowTraceInterventionRunner


def test_state_off_changes_selected_state_only():
    model = FlowTracePMTModel(fine_dim=8, coarse_dim=16, dense_dim=16, state_dim=8, num_tracks=2, num_states=3)
    b = model(torch.randn(1, 4, 16), torch.randn(1, 8, 3, 4, 4), torch.randn(1, 16, 3, 2, 2))
    before = b.state_memory.clone()
    out = FlowTraceInterventionRunner().apply(b, {"type": "state_off", "state_idx": 1})
    assert torch.all(out.state_memory[:, 1] == 0)
    assert torch.allclose(out.state_memory[:, 0], before[:, 0])
