import torch

from fate_x.losses.acpr_flowcal_v2_losses import masked_language_model_loss, control_rmse_loss
from fate_x.engine.evaluate_v51_event_metrics import compute_text_cider_proxy


def test_v2_losses_are_finite_and_differentiable():
    logits = torch.randn(2, 5, 11, requires_grad=True)
    labels = torch.randint(0, 11, (2, 3))
    pos = torch.tensor([[0, 2, 4], [1, 3, 4]])
    mlm = masked_language_model_loss(logits, labels, pos)
    pred = torch.randn(2, 32, 2, requires_grad=True)
    target = torch.randn(2, 2, 32)
    ctrl = control_rmse_loss(pred, target)
    loss = mlm + ctrl
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert pred.grad is not None


def test_cider_proxy_prefers_exact_text():
    exact = compute_text_cider_proxy(["car slows down"], ["car slows down"])
    wrong = compute_text_cider_proxy(["car speeds up"], ["car slows down"])
    assert exact > wrong
