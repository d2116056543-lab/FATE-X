from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from src.modeling.load_swin import get_swin_model, myVideoSwin

from .types import SwinBackboneOutput


def _swin_args_from_config(cfg: dict[str, Any] | None) -> SimpleNamespace:
    raw = cfg or {}
    data = raw.get("data", {})
    backbone = raw.get("model", {}).get("backbone", {})
    paths = raw.get("paths", {})
    return SimpleNamespace(
        img_res=int(data.get("image_resolution", backbone.get("input_resolution", 224))),
        vidswin_size="base",
        kinetics=600,
        pretrained_2d=False,
        pretrained_checkpoint="",
        grid_feat=True,
        reload_pretrained_swin=False,
        use_checkpoint=False,
        freeze_backbone=False,
        video_swin_kinetics_checkpoint=paths.get("video_swin_kinetics_checkpoint"),
    )


class ACPRDynFlowSwinBackbone(nn.Module):
    """Repository Video Swin-B wrapper for the formal direct-image path.

    Inputs use the formal batch layout ``[B, 32, 3, 224, 224]``. The wrapped
    ADAPT Video Swin path expects ``[B, 3, T, H, W]`` and returns native stage
    tensors through ``return_stages=(2, 3)``. No extra Video Swin pass is made.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        self.forward_count = 0
        self.args = _swin_args_from_config(cfg)
        self.video_swin: myVideoSwin = get_swin_model(self.args)

    def forward(self, frames: Tensor) -> SwinBackboneOutput:
        if frames.ndim != 5 or frames.shape[1] != 32 or frames.shape[2] != 3:
            raise ValueError(f"frames must be [B,32,3,H,W], got {tuple(frames.shape)}")
        self.forward_count += 1
        x = frames.permute(0, 2, 1, 3, 4)
        final_tokens, predicate_stage, final_stage = self.video_swin(x, return_stages=(2, 3))
        predicate = predicate_stage.permute(0, 2, 3, 4, 1).contiguous()
        final = final_stage.permute(0, 2, 3, 4, 1).contiguous()
        temporal = final.mean(dim=(2, 3))
        dense = final.reshape(final.shape[0], final.shape[1] * final.shape[2] * final.shape[3], final.shape[-1])
        return SwinBackboneOutput(
            predicate_grid=predicate,
            final_grid=final,
            temporal_global=temporal,
            dense_final_tokens=dense,
            forward_count=self.forward_count,
        )
