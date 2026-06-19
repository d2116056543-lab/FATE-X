import torch

from fate_x.acpr_flow.online_reason_target import build_action_residual_reason_target


def test_online_reason_target_residualizes_action_direction_and_detaches():
    emb = torch.randn(3, 6, 12, requires_grad=True)
    action_mask = torch.zeros(3, 6, dtype=torch.bool)
    reason_mask = torch.zeros(3, 6, dtype=torch.bool)
    action_mask[:, :2] = True
    reason_mask[:, 2:] = True
    target = build_action_residual_reason_target(emb, action_mask, reason_mask)
    assert target.shape == (3, 12)
    assert target.requires_grad is False
    assert torch.isfinite(target).all()
