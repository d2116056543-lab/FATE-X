from __future__ import annotations

import torch


def traffic_factor_control_correlation(traffic_factor: torch.Tensor, control_pred: torch.Tensor) -> dict[str, float]:
    tf = traffic_factor.detach().flatten()
    cp = control_pred.detach().mean(dim=-1).flatten()
    n = min(tf.numel(), cp.numel())
    if n < 2:
        return {"pred_control_corr": None}
    tf = tf[:n] - tf[:n].mean()
    cp = cp[:n] - cp[:n].mean()
    denom = tf.norm() * cp.norm()
    if float(denom) == 0.0:
        return {"pred_control_corr": None}
    return {"pred_control_corr": float((tf * cp).sum() / denom)}
