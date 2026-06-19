import torch

from fate_x.acpr_flow.model import ACPRFlowModel


def test_e2e_direct_image_smoke_backward_uses_32_frames_224():
    model = ACPRFlowModel()
    frames = torch.randn(1, 32, 3, 224, 224)
    out = model(frames)
    assert out.bundle.fine_grid.shape[1] == 32
    out.total_loss.backward()
    assert model.predicate_field.queries.grad.abs().sum() > 0
