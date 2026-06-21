from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

from fate_x.losses.acpr_flowcal_losses import (
    control_signal_error,
    is_circular_control_signal,
    signed_circular_delta_deg,
)


def _safe_float(value: Tensor | float) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _decision_from_delta(delta: Tensor, threshold: float, negative_name: str, neutral_name: str, positive_name: str) -> tuple[Tensor, list[str]]:
    labels = torch.ones_like(delta, dtype=torch.long)
    labels = torch.where(delta < -float(threshold), torch.zeros_like(labels), labels)
    labels = torch.where(delta > float(threshold), torch.full_like(labels, 2), labels)
    return labels, [negative_name, neutral_name, positive_name]


def _classification_metrics(pred_labels: Tensor, target_labels: Tensor, label_names: list[str]) -> dict[str, Any]:
    valid = target_labels.ge(0)
    if not bool(valid.any()):
        return {"available": False, "label_names": label_names, "sample_count": 0}
    pred = pred_labels[valid].detach().cpu()
    target = target_labels[valid].detach().cpu()
    correct = pred.eq(target)
    per_class: dict[str, dict[str, Any]] = {}
    recalls = []
    for idx, name in enumerate(label_names):
        mask = target.eq(idx)
        support = int(mask.sum().item())
        if support > 0:
            recall = float(correct[mask].float().mean().item())
            recalls.append(recall)
        else:
            recall = None
        per_class[name] = {"support": support, "recall": recall}
    return {
        "available": True,
        "label_names": label_names,
        "sample_count": int(target.numel()),
        "accuracy": float(correct.float().mean().item()),
        "macro_recall": float(sum(recalls) / max(len(recalls), 1)) if recalls else None,
        "per_class": per_class,
    }


def compute_control_metrics(
    pred: Tensor,
    target: Tensor,
    *,
    signal_names: Sequence[str] = ("course", "speed"),
    invalid_value: float = -1.0,
    thresholds: Sequence[float] = (0.1, 0.5, 1.0, 5.0, 10.0),
    speed_delta_threshold: float = 0.5,
    course_delta_threshold: float = 0.05,
    include_decision_proxy: bool = False,
) -> dict[str, Any]:
    """Compute ADAPT-style continuous control quality.

    pred and target must be shaped [B, T, S], where signal order follows signal_names.
    Invalid target values are ignored elementwise. Coarse speed/course decision proxies
    are diagnostic only and are excluded unless include_decision_proxy=True.
    """
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(f"control metrics expect [B,T,S], got pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if pred.shape[-1] != len(signal_names):
        raise ValueError(f"signal_names length {len(signal_names)} does not match signal dim {pred.shape[-1]}")
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    finite = torch.isfinite(pred) & torch.isfinite(target)
    valid = finite & target.ne(float(invalid_value))
    error = control_signal_error(pred, target, signal_names=signal_names)
    out: dict[str, Any] = {
        "metric_family": "adapt_continuous_control",
        "metric_name": "control_rmse_threshold_accuracy",
        "diagnostic_decision_proxy_available": bool(include_decision_proxy),
        "sample_count": int(pred.shape[0]),
        "time_steps": int(pred.shape[1]),
        "signal_names": list(signal_names),
        "valid_value_count": int(valid.sum().item()),
        "signals": {},
    }
    for idx, name in enumerate(signal_names):
        mask = valid[..., idx]
        if bool(mask.any()):
            sig_err = error[..., idx][mask]
            sig_abs = sig_err.abs()
            sig = {
                "valid_count": int(mask.sum().item()),
                "rmse": _safe_float(torch.sqrt(torch.mean(sig_err.pow(2)))),
                "mae": _safe_float(torch.mean(sig_abs)),
            }
            for threshold in thresholds:
                sig[f"acc_at_{threshold:g}"] = _safe_float(sig_abs.lt(float(threshold)).float().mean())
        else:
            sig = {"valid_count": 0, "rmse": None, "mae": None}
            for threshold in thresholds:
                sig[f"acc_at_{threshold:g}"] = None
        out["signals"][str(name)] = sig

    if not include_decision_proxy:
        return out

    def first_last_delta(values: Tensor, valid_mask: Tensor, signal_idx: int, signal_name: str) -> tuple[Tensor, Tensor]:
        deltas = []
        keep = []
        for b in range(values.shape[0]):
            idxs = torch.nonzero(valid_mask[b, :, signal_idx], as_tuple=False).flatten()
            if idxs.numel() >= 2:
                delta = values[b, idxs[-1], signal_idx] - values[b, idxs[0], signal_idx]
                if is_circular_control_signal(signal_name):
                    delta = signed_circular_delta_deg(delta)
                deltas.append(delta)
                keep.append(True)
            else:
                deltas.append(torch.tensor(0.0))
                keep.append(False)
        return torch.stack(deltas), torch.tensor(keep, dtype=torch.bool)

    name_to_idx = {name: i for i, name in enumerate(signal_names)}
    if "speed" in name_to_idx:
        idx = name_to_idx["speed"]
        pred_delta, pred_keep = first_last_delta(pred, valid, idx, "speed")
        target_delta, target_keep = first_last_delta(target, valid, idx, "speed")
        keep = pred_keep & target_keep
        pred_labels, label_names = _decision_from_delta(pred_delta, speed_delta_threshold, "decrease", "maintain", "increase")
        target_labels, _ = _decision_from_delta(target_delta, speed_delta_threshold, "decrease", "maintain", "increase")
        target_labels = torch.where(keep, target_labels, torch.full_like(target_labels, -1))
        out["speed_decision"] = _classification_metrics(pred_labels, target_labels, label_names)
    if "course" in name_to_idx:
        idx = name_to_idx["course"]
        pred_delta, pred_keep = first_last_delta(pred, valid, idx, "course")
        target_delta, target_keep = first_last_delta(target, valid, idx, "course")
        keep = pred_keep & target_keep
        pred_labels, label_names = _decision_from_delta(pred_delta, course_delta_threshold, "left", "straight", "right")
        target_labels, _ = _decision_from_delta(target_delta, course_delta_threshold, "left", "straight", "right")
        target_labels = torch.where(keep, target_labels, torch.full_like(target_labels, -1))
        out["course_decision"] = _classification_metrics(pred_labels, target_labels, label_names)
    return out
