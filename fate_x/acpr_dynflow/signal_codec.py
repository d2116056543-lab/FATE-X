from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass
class BDDSignalCodec:
    signal_names: tuple[str, ...] = ("course", "speed")
    invalid_value: float = -1.0
    mean: Tensor | None = None
    std: Tensor | None = None

    def fit(self, values: Tensor) -> "BDDSignalCodec":
        valid = torch.isfinite(values) & values.ne(float(self.invalid_value))
        mean = []
        std = []
        for idx in range(values.shape[-1]):
            mask = valid[..., idx]
            x = values[..., idx][mask]
            if x.numel() == 0:
                mean.append(torch.tensor(0.0, device=values.device))
                std.append(torch.tensor(1.0, device=values.device))
            else:
                mean.append(x.mean())
                std.append(x.std().clamp_min(1e-6))
        self.mean = torch.stack(mean).detach()
        self.std = torch.stack(std).detach()
        return self

    def _stats(self, device: torch.device) -> tuple[Tensor, Tensor]:
        mean = self.mean if self.mean is not None else torch.zeros(len(self.signal_names))
        std = self.std if self.std is not None else torch.ones(len(self.signal_names))
        return mean.to(device), std.to(device).clamp_min(1e-6)

    def encode(self, raw: Tensor) -> Tensor:
        mean, std = self._stats(raw.device)
        encoded = (raw - mean) / std
        return torch.where(raw.eq(float(self.invalid_value)), raw, encoded)

    def decode(self, normalized: Tensor) -> Tensor:
        mean, std = self._stats(normalized.device)
        return normalized * std + mean

    def valid_mask(self, target: Tensor) -> Tensor:
        return torch.isfinite(target) & target.ne(float(self.invalid_value))

    def official_metrics(self, pred_raw: Tensor, target_raw: Tensor, thresholds: Sequence[float] = (0.1, 0.5, 1.0, 5.0, 10.0)) -> dict[str, Any]:
        if pred_raw.shape != target_raw.shape:
            raise ValueError(f"shape mismatch pred={tuple(pred_raw.shape)} target={tuple(target_raw.shape)}")
        valid = self.valid_mask(target_raw) & torch.isfinite(pred_raw)
        out: dict[str, Any] = {"metric_family": "adapt_continuous_control", "signal_names": list(self.signal_names), "signals": {}}
        err = pred_raw.float() - target_raw.float()
        for idx, name in enumerate(self.signal_names):
            mask = valid[..., idx]
            if bool(mask.any()):
                e = err[..., idx][mask]
                ae = e.abs()
                sig = {"rmse": float(torch.sqrt(e.pow(2).mean()).item()), "mae": float(ae.mean().item()), "valid_count": int(mask.sum().item())}
                for th in thresholds:
                    sig[f"acc_at_{th:g}"] = float(ae.lt(float(th)).float().mean().item())
            else:
                sig = {"rmse": None, "mae": None, "valid_count": 0}
                for th in thresholds:
                    sig[f"acc_at_{th:g}"] = None
            out["signals"][name] = sig
        return out

