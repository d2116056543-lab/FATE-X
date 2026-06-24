from __future__ import annotations

import pytest
import torch

from fate_x.losses.acpr_dynflow_swin_losses import benefit_gate_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_binary_losses_are_safe_under_cuda_bf16_autocast():
    logits = torch.randn(2, 32, 2, device="cuda", requires_grad=True)
    gate = torch.sigmoid(logits)
    target = torch.rand(2, 32, 2, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = benefit_gate_loss(gate, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
