from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import VideoBackboneOutput


def _infer_latent_dim(video_swin: nn.Module, fallback: int = 1024) -> int:
    norm = getattr(getattr(video_swin, "backbone", None), "norm", None)
    shape = getattr(norm, "normalized_shape", None)
    if shape:
        return int(shape[0])
    return int(fallback)


def _build_real_video_swin(image_resolution: int = 224) -> nn.Module:
    from src.modeling.load_swin import get_swin_model

    args = SimpleNamespace(
        img_res=int(image_resolution),
        vidswin_size="base",
        kinetics=600,
        pretrained_2d=False,
        pretrained_checkpoint="",
        grid_feat=True,
    )
    return get_swin_model(args)


class _NonFormalTinyVideoSwin(nn.Module):
    """Small direct-image fallback for unit tests only.

    Formal YAML sets ``use_real_video_swin=True`` and therefore never uses this
    module. Keeping this path explicit prevents tests from silently pretending
    that the fallback is an ADAPT-equivalent backbone.
    """

    def __init__(self, latent_dim: int = 1024):
        super().__init__()
        self.proj = nn.Conv3d(3, latent_dim, kernel_size=(1, 16, 16), stride=(1, 16, 16))
        self.backbone = type("Backbone", (), {"norm": type("Norm", (), {"normalized_shape": (latent_dim,)})()})()

    def forward(self, x: Tensor, return_stages: bool = False):
        fine = self.proj(x.float())
        coarse = fine[:, :, :, ::2, ::2]
        return (fine, fine, coarse) if return_stages else fine


def _temporal_align(grid: Tensor, frames: int = 32) -> Tensor:
    if grid.shape[1] == frames:
        return grid
    b, t, h, w, d = grid.shape
    x = grid.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, t)
    x = F.interpolate(x, size=frames, mode="linear", align_corners=False)
    return x.reshape(b, h, w, d, frames).permute(0, 4, 1, 2, 3)


def _project_dense_tokens(tokens: Tensor, projector: nn.Module) -> Tensor:
    return projector(tokens)


def _extract_native_stages(result: Any) -> Tuple[Tensor, Tensor]:
    """Return fine/coarse native Video-Swin grids without replacing ADAPT stages.

    ADAPT Video-Swin returns either a final grid or a tuple containing the final
    grid plus intermediate native stages when ``return_stages=True``. FlowCal V2
    needs the native fine/coarse tensors for its transport path, so this helper
    preserves the released backbone outputs and only falls back to the final grid
    for non-formal tiny unit-test backbones.
    """
    if isinstance(result, tuple):
        final = result[0]
        stages = list(result[1:])
        fine = stages[-2] if len(stages) >= 2 else final
        coarse = stages[-1] if stages else final
        return fine, coarse
    return result, result


def _fuse_reasoning_grids(fine: Tensor, coarse: Tensor) -> Tensor:
    b, t, hf, wf, d = fine.shape
    c = coarse.permute(0, 1, 4, 2, 3).reshape(b * t, d, coarse.shape[2], coarse.shape[3])
    c = F.interpolate(c, size=(hf, wf), mode="bilinear", align_corners=False).reshape(b, t, d, hf, wf).permute(0, 1, 3, 4, 2)
    return F.layer_norm(fine + c, (d,))


class ADAPTVideoBackboneV2(nn.Module):
    def __init__(
        self,
        output_dim: int = 256,
        adapt_feature_dim: int = 512,
        video_swin: Optional[nn.Module] = None,
        latent_dim: Optional[int] = None,
        adapt_checkpoint: Optional[str] = None,
        use_real_video_swin: bool = False,
        image_resolution: int = 224,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.adapt_feature_dim = int(adapt_feature_dim)
        self.video_swin = video_swin if video_swin is not None else (_build_real_video_swin(image_resolution) if use_real_video_swin else _NonFormalTinyVideoSwin())
        self.latent_dim = int(latent_dim or _infer_latent_dim(self.video_swin))
        self.fc = nn.Linear(self.latent_dim, self.adapt_feature_dim)
        self.reason_proj = nn.Linear(self.adapt_feature_dim, output_dim)
        self.forward_count = 0
        self.load_report: Dict[str, Any] = {
            "video_swin": {
                "loaded": bool(use_real_video_swin or video_swin is not None),
                "formal": bool(use_real_video_swin),
                "source": "injected" if video_swin is not None else ("src.modeling.load_swin.get_swin_model" if use_real_video_swin else "non_formal_unit_test_fallback"),
            },
            "adapt_fc": {"loaded": False, "checkpoint": None, "missing": ["fc.weight", "fc.bias"]},
        }
        if adapt_checkpoint:
            self.load_adapt_fc(adapt_checkpoint)

    def load_adapt_fc(self, checkpoint_path: str) -> Dict[str, Any]:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("model", state) if isinstance(state, dict) else {}
        weight = state_dict.get("fc.weight")
        bias = state_dict.get("fc.bias")
        report: Dict[str, Any] = {"loaded": False, "checkpoint": str(checkpoint_path), "missing": [], "unexpected": []}
        if weight is None or bias is None:
            report["missing"] = [name for name, value in (("fc.weight", weight), ("fc.bias", bias)) if value is None]
        elif tuple(weight.shape) != tuple(self.fc.weight.shape) or tuple(bias.shape) != tuple(self.fc.bias.shape):
            report["unexpected"] = [f"fc.weight{tuple(weight.shape)} expected {tuple(self.fc.weight.shape)}", f"fc.bias{tuple(bias.shape)} expected {tuple(self.fc.bias.shape)}"]
        else:
            with torch.no_grad():
                self.fc.weight.copy_(weight)
                self.fc.bias.copy_(bias)
            report["loaded"] = True
        self.load_report["adapt_fc"] = report
        return report

    def reset_forward_counter(self) -> None:
        self.forward_count = 0

    def _run_video_swin(self, frames: Tensor) -> Tuple[Tensor, Tensor]:
        images = frames.permute(0, 2, 1, 3, 4)
        result = self.video_swin(images, return_stages=True)
        return _extract_native_stages(result)

    def _to_bt_hw_c(self, grid: Tensor) -> Tensor:
        if grid.dim() != 5:
            raise ValueError(f"expected Video-Swin grid [B,C,T,H,W], got {tuple(grid.shape)}")
        return grid.permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, frames: Tensor) -> VideoBackboneOutput:
        self.forward_count += 1
        fine_raw_bcthw, coarse_raw_bcthw = self._run_video_swin(frames)
        fine_raw = self._to_bt_hw_c(fine_raw_bcthw)
        coarse_raw = self._to_bt_hw_c(coarse_raw_bcthw)
        fine_adapt = self.fc(fine_raw)
        coarse_adapt = self.fc(coarse_raw)
        fine = self.reason_proj(fine_adapt)
        coarse = self.reason_proj(coarse_adapt)
        fine_aligned = _temporal_align(fine, frames=frames.shape[1])
        coarse_aligned = _temporal_align(coarse, frames=frames.shape[1])
        fused_grid = _fuse_reasoning_grids(fine_aligned, coarse_aligned)
        dense_raw = fine_raw.reshape(
            frames.shape[0],
            fine_raw.shape[1] * fine_raw.shape[2] * fine_raw.shape[3],
            fine_raw.shape[-1],
        )
        dense_projected = _project_dense_tokens(dense_raw, self.fc)
        return VideoBackboneOutput(
            fine_native=fine,
            coarse_native=coarse,
            fine_aligned=fine_aligned,
            coarse_aligned=coarse_aligned,
            fused_grid=fused_grid,
            dense_tokens_raw=dense_raw,
            dense_tokens_projected=dense_projected,
            forward_count=self.forward_count,
        )


ADAPTVideoBackbone = ADAPTVideoBackboneV2
