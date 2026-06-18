import torch

from fate_x.losses.flowtrace_losses import FlowTraceLoss
from fate_x.models.flowtrace_pmt_model import FlowTracePMTModel


def test_intervention_loss_is_real_and_marked_available():
    model = FlowTracePMTModel(fine_dim=8, coarse_dim=16, dense_dim=16, state_dim=8, num_tracks=2, num_states=3)
    bundle = model(
        torch.randn(2, 4, 16),
        torch.randn(2, 8, 3, 4, 4),
        torch.randn(2, 16, 3, 2, 2),
    )

    _, logs = FlowTraceLoss(state_dim=8)(bundle)

    assert logs["intervention_available"].item() == 1.0
    assert torch.isfinite(logs["intervention"])
    assert logs["intervention"].item() > 0.0
