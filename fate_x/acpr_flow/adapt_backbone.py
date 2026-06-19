from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class BackboneLoadReport:
    requested_checkpoint: str
    checkpoint_exists: bool
    loaded: bool
    reason: str = ""


class _FormalBackboneFallback(nn.Module):
    """Non-production lightweight path used only when checkpoint loading is disabled."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.fine = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(64, state_dim, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
        )
        self.coarse = nn.Sequential(
            nn.Conv2d(state_dim, state_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        fine = self.fine(x)
        coarse = self.coarse(fine)
        dense = coarse.flatten(2).transpose(1, 2).reshape(b, t * coarse.shape[-2] * coarse.shape[-1], -1)
        return fine.reshape(b, t, fine.shape[1], fine.shape[2], fine.shape[3]), coarse.reshape(
            b, t, coarse.shape[1], coarse.shape[2], coarse.shape[3]
        ), dense


class ADAPTVideoSwinMultiscaleBackbone(nn.Module):
    """Formal ACPR backbone wrapper around ADAPT Video Swin multiscale features."""

    formal_backbone_name = "adapt_video_swin_multiscale"

    def __init__(
        self,
        state_dim: int = 256,
        image_resolution: int = 224,
        fine_stage: int = 2,
        coarse_stage: int = 3,
        load_pretrained: bool = True,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.image_resolution = int(image_resolution)
        self.fine_stage = int(fine_stage)
        self.coarse_stage = int(coarse_stage)
        self.video_swin: nn.Module | None = None
        self.fallback = _FormalBackboneFallback(state_dim)
        requested = checkpoint_path or "models/video_swin_transformer/swin_base_patch244_window877_kinetics600_22k.pth"
        exists = Path(requested).exists()
        self.load_report = BackboneLoadReport(
            requested_checkpoint=requested,
            checkpoint_exists=exists,
            loaded=False,
            reason="load_pretrained_backbone=false",
        )
        if load_pretrained:
            if not exists:
                raise FileNotFoundError(f"Video Swin checkpoint not found: {requested}")
            from src.modeling.load_swin import get_swin_model

            args = SimpleNamespace(
                img_res=self.image_resolution,
                vidswin_size="base",
                kinetics=600,
                pretrained_2d=False,
                pretrained_checkpoint="",
                grid_feat=True,
            )
            self.video_swin = get_swin_model(args)
            self.load_report = BackboneLoadReport(
                requested_checkpoint=requested,
                checkpoint_exists=True,
                loaded=True,
                reason="loaded_by_src.modeling.load_swin.get_swin_model",
            )
        self.fine_proj = nn.LazyLinear(self.state_dim)
        self.coarse_proj = nn.LazyLinear(self.state_dim)
        self.fuse_norm = nn.LayerNorm(self.state_dim)

    @staticmethod
    def _stage_to_btchw(stage: Tensor, batch: int, frames: int) -> Tensor:
        if stage.ndim == 5:
            # Video Swin emits [B, C, T', H, W].
            return stage.permute(0, 2, 1, 3, 4).contiguous()
        if stage.ndim == 3:
            # Token output [B, N, C]; treat as one time step grid if needed.
            n = stage.shape[1]
            side = int(n ** 0.5)
            if side * side == n:
                return stage.transpose(1, 2).reshape(batch, 1, stage.shape[-1], side, side)
        raise ValueError(f"Unsupported Video Swin stage shape: {tuple(stage.shape)}")

    @staticmethod
    def _to_bthwc(stage_btchw: Tensor) -> Tensor:
        return stage_btchw.permute(0, 1, 3, 4, 2).contiguous()

    def _run_video_swin(self, frames: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.video_swin is None:
            return self.fallback(frames)
        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        outputs = self.video_swin(x, return_stages=True)
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            raise RuntimeError("Video Swin return_stages=True did not return multiscale stages")
        dense_tokens = outputs[0]
        stages = outputs[1:]
        fine_raw = stages[min(self.fine_stage, len(stages) - 1)]
        coarse_raw = stages[min(self.coarse_stage, len(stages) - 1)]
        fine = self._stage_to_btchw(fine_raw, frames.shape[0], frames.shape[1])
        coarse = self._stage_to_btchw(coarse_raw, frames.shape[0], frames.shape[1])
        if dense_tokens.ndim != 3:
            dense_tokens = coarse.flatten(3).permute(0, 1, 3, 2).reshape(frames.shape[0], -1, coarse.shape[2])
        return fine, coarse, dense_tokens

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        if frames.ndim != 5:
            raise ValueError(f"ACPR expects direct frames [B,T,3,H,W], got {tuple(frames.shape)}")
        if frames.shape[1] != 32 or frames.shape[2] != 3:
            raise ValueError(f"ACPR formal path expects [B,32,3,H,W], got {tuple(frames.shape)}")
        fine_btchw, coarse_btchw, dense_tokens = self._run_video_swin(frames)
        fine = self._to_bthwc(fine_btchw)
        coarse = self._to_bthwc(coarse_btchw)
        fine_proj = self.fine_proj(fine)
        coarse_proj = self.coarse_proj(coarse)
        b, t, hc, wc, d = coarse_proj.shape
        coarse_up = F.interpolate(
            coarse_proj.reshape(b * t, hc, wc, d).permute(0, 3, 1, 2),
            size=fine_proj.shape[2:4],
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape_as(fine_proj)
        fused = self.fuse_norm(fine_proj + coarse_up)
        return {
            "fine_grid": fine_proj,
            "coarse_grid": coarse_proj,
            "fused_grid": fused,
            "dense_tokens": dense_tokens,
        }
