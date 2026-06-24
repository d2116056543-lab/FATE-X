from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass
class BDDXSignalCodec:
    signal_names: tuple[str, ...] = ("course", "speed")
    invalid_value: float = -1.0
    mean: Tensor | None = None
    std: Tensor | None = None

    def fit(self, values: Tensor) -> "BDDXSignalCodec":
        valid = torch.isfinite(values) & values.ne(float(self.invalid_value))
        means = []
        stds = []
        for index in range(values.shape[-1]):
            mask = valid[..., index]
            observed = values[..., index][mask]
            if observed.numel() == 0:
                means.append(torch.tensor(0.0, device=values.device))
                stds.append(torch.tensor(1.0, device=values.device))
            else:
                means.append(observed.mean())
                stds.append(observed.std(unbiased=False).clamp_min(1e-6))
        self.mean = torch.stack(means).detach()
        self.std = torch.stack(stds).detach()
        return self

    def _stats(self, device: torch.device) -> tuple[Tensor, Tensor]:
        mean = self.mean if self.mean is not None else torch.zeros(len(self.signal_names))
        std = self.std if self.std is not None else torch.ones(len(self.signal_names))
        return mean.to(device), std.to(device).clamp_min(1e-6)

    def valid_mask(self, target: Tensor) -> Tensor:
        return torch.isfinite(target) & target.ne(float(self.invalid_value))

    def encode(self, raw: Tensor) -> Tensor:
        mean, std = self._stats(raw.device)
        normalized = (raw - mean) / std
        return torch.where(raw.eq(float(self.invalid_value)), raw, normalized)

    def decode(self, normalized: Tensor) -> Tensor:
        mean, std = self._stats(normalized.device)
        return normalized * std + mean

    def official_metrics(
        self,
        pred_raw: Tensor,
        target_raw: Tensor,
        thresholds: Sequence[float] = (0.1, 0.5, 1.0, 5.0, 10.0),
    ) -> dict[str, Any]:
        if pred_raw.shape != target_raw.shape:
            raise ValueError(f"shape mismatch pred={tuple(pred_raw.shape)} target={tuple(target_raw.shape)}")
        valid = self.valid_mask(target_raw) & torch.isfinite(pred_raw)
        error = pred_raw.float() - target_raw.float()
        metrics: dict[str, Any] = {
            "metric_family": "adapt_continuous_control",
            "signal_names": list(self.signal_names),
            "signals": {},
        }
        for index, name in enumerate(self.signal_names):
            mask = valid[..., index]
            signal: dict[str, Any]
            if bool(mask.any()):
                e = error[..., index][mask]
                ae = e.abs()
                signal = {
                    "RMSE": float(torch.sqrt(e.square().mean()).item()),
                    "MAE": float(ae.mean().item()),
                    "valid_count": int(mask.sum().item()),
                }
                for threshold in thresholds:
                    signal[f"Acc@{threshold:g}"] = float(ae.le(float(threshold)).float().mean().item())
            else:
                signal = {"RMSE": None, "MAE": None, "valid_count": 0}
                for threshold in thresholds:
                    signal[f"Acc@{threshold:g}"] = None
            metrics["signals"][name] = signal
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "signal_names": list(self.signal_names),
            "invalid_value": self.invalid_value,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "BDDXSignalCodec":
        codec = cls(tuple(state.get("signal_names", ("course", "speed"))), float(state.get("invalid_value", -1.0)))
        if state.get("mean") is not None:
            codec.mean = torch.tensor(state["mean"], dtype=torch.float32)
        if state.get("std") is not None:
            codec.std = torch.tensor(state["std"], dtype=torch.float32)
        return codec
