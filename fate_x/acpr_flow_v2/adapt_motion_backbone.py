from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.modeling.load_sensor_pred_head import Sensor_Pred_Head


def _default_sensor_args(input_dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        img_feature_dim=int(input_dim),
        grid_feat=True,
        config_name="",
        model_name_or_path="models/captioning/bert-base-uncased",
        signal_types=["course", "speed"],
    )


class ADAPTMotionBackbone(nn.Module):
    """V2 wrapper around ADAPT's released sensor prediction head."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 768, output_dim: int = 2, args: Optional[Any] = None):
        super().__init__()
        if output_dim != 2:
            raise ValueError("BDD-X ADAPT control head predicts exactly [course, speed]")
        self.input_dim = int(input_dim)
        self.expected_hidden_dim = int(hidden_dim)
        self.sensor_head = Sensor_Pred_Head(args or _default_sensor_args(self.input_dim))
        self.sensor_head.eval()
        self.load_report: Dict[str, Any] = {"loaded": [], "missing": [], "unexpected": []}

    @classmethod
    def from_adapt_checkpoint(cls, checkpoint_path: Optional[str] = None, input_dim: int = 256, hidden_dim: int = 768) -> "ADAPTMotionBackbone":
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim)
        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            state_dict = state.get("model", state) if isinstance(state, dict) else {}
            own = model.sensor_head.state_dict()
            filtered = {}
            unexpected = []
            for key, value in state_dict.items():
                stripped = key
                for prefix in ("sensor_pred_head.", "module.sensor_pred_head."):
                    if stripped.startswith(prefix):
                        stripped = stripped[len(prefix):]
                if stripped in own and tuple(own[stripped].shape) == tuple(value.shape):
                    filtered[stripped] = value
                elif "sensor_pred_head" in key:
                    unexpected.append(key)
            result = model.sensor_head.load_state_dict(filtered, strict=False)
            model.load_report = {
                "checkpoint": str(checkpoint_path),
                "loaded": sorted(filtered.keys()),
                "missing": sorted(result.missing_keys),
                "unexpected": sorted(unexpected + list(result.unexpected_keys)),
            }
        return model

    def encode(self, dense_tokens: Tensor) -> Tensor:
        return self.sensor_head.encode(dense_tokens.float())

    def predict(self, dense_tokens: Tensor, steps: int = 32) -> Tuple[Tensor, Tensor]:
        hidden = self.encode(dense_tokens)
        if hidden.shape[1] != steps:
            hidden_t = hidden.transpose(1, 2)
            hidden = F.interpolate(hidden_t, size=steps, mode="linear", align_corners=False).transpose(1, 2)
        return self.sensor_head.decoder(hidden), hidden


def from_adapt_checkpoint(*args, **kwargs) -> ADAPTMotionBackbone:
    return ADAPTMotionBackbone.from_adapt_checkpoint(*args, **kwargs)


ADAPTMotionTransformer = ADAPTMotionBackbone
