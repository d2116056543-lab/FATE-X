from __future__ import annotations

import torch


def traffic_flow_utilization(full_rmse: torch.Tensor, off_rmse: torch.Tensor) -> torch.Tensor:
    return off_rmse - full_rmse

