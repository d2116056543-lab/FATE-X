from __future__ import annotations

import torch
from torch import Tensor


def split_action_reason_loss(token_losses: Tensor, token_type_ids: Tensor) -> dict[str, Tensor]:
    action = token_losses[token_type_ids == 0]
    reason = token_losses[token_type_ids == 1]
    return {
        "action_loss": action.mean() if action.numel() else token_losses.new_zeros(()),
        "reason_loss": reason.mean() if reason.numel() else token_losses.new_zeros(()),
    }
