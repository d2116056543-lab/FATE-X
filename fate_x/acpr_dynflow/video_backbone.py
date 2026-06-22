from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import DynFlowBackboneOutput


def _convert_official_vswin_to_torchvision(state_dict: dict[str, Tensor], target: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Convert official Video-Swin `backbone.*` keys to torchvision `swin3d_b`.

    The official K600 checkpoint and torchvision model have matching tensor
    shapes for the backbone, but use different module names.
    """

    converted: dict[str, Tensor] = {}
    layer_to_feature = {0: 0, 1: 2, 2: 4, 3: 6}
    downsample_to_feature = {0: 1, 1: 3, 2: 5}
    skipped: list[str] = []

    for raw_key, value in state_dict.items():
        key = raw_key
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        elif key.startswith("cls_head."):
            skipped.append(raw_key)
            continue

        if key.startswith("patch_embed."):
            mapped = key
        elif key.startswith("norm."):
            mapped = key
        elif key.startswith("layers."):
            parts = key.split(".")
            if len(parts) < 3:
                skipped.append(raw_key)
                continue
            layer_idx = int(parts[1])
            rest = ".".join(parts[2:])
            if rest.startswith("blocks."):
                block_rest = rest[len("blocks.") :]
                block_id, _, tail = block_rest.partition(".")
                mapped = f"features.{layer_to_feature[layer_idx]}.{block_id}.{tail}"
            elif rest.startswith("downsample.") and layer_idx in downsample_to_feature:
                tail = rest[len("downsample.") :]
                mapped = f"features.{downsample_to_feature[layer_idx]}.{tail}"
            else:
                skipped.append(raw_key)
                continue
            mapped = mapped.replace(".mlp.fc1.", ".mlp.0.")
            mapped = mapped.replace(".mlp.fc2.", ".mlp.3.")
        else:
            skipped.append(raw_key)
            continue

        if mapped in target and tuple(target[mapped].shape) == tuple(value.shape):
            converted[mapped] = value
        else:
            skipped.append(raw_key)

    report = {
        "source_keys": len(state_dict),
        "converted_keys": len(converted),
        "target_keys": len(target),
        "skipped_keys": len(skipped),
        "converted_ratio": float(len(converted) / max(1, len(target))),
        "sample_skipped": skipped[:20],
    }
    return converted, report


class ACPRDynFlowVideoBackbone(nn.Module):
    """Direct-image Video Swin backbone for ACPR-DynFlow.

    Formal config uses torchvision's actual `swin3d_b` architecture and loads
    the official Kinetics-600 Video-Swin checkpoint via deterministic key
    conversion. A tiny fallback is kept only for unit tests without assets.
    """

    def __init__(self, state_dim: int = 256, text_dim: int = 768, checkpoint_path: str | None = None):
        super().__init__()
        self.forward_count = 0
        self.kinetics_checkpoint_path = checkpoint_path or ""
        self.kinetics_loaded = False
        self.kinetics_load_report: dict[str, Any] = {}
        self._uses_torchvision_swin = False

        checkpoint = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint is not None and checkpoint.exists():
            from torchvision.models.video import swin3d_b

            self.swin = swin3d_b(weights=None)
            raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
            sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
            converted, report = _convert_official_vswin_to_torchvision(sd, self.swin.state_dict())
            missing, unexpected = self.swin.load_state_dict(converted, strict=False)
            report["missing_keys"] = len(missing)
            report["unexpected_keys"] = len(unexpected)
            report["sample_missing"] = list(missing)[:20]
            report["sample_unexpected"] = list(unexpected)[:20]
            self.kinetics_load_report = report
            self.kinetics_loaded = report["converted_keys"] >= 300
            self._uses_torchvision_swin = True
            self.state_proj = nn.Linear(1024, state_dim)
            self.text_proj = nn.Linear(state_dim, text_dim)
            # Plan freezes early stages and trains stages 2/3.
            for p in self.swin.patch_embed.parameters():
                p.requires_grad = False
            for idx in [0, 1, 2, 3]:
                for p in self.swin.features[idx].parameters():
                    p.requires_grad = False
        else:
            self.stage0 = nn.Conv2d(3, 32, 3, padding=1)
            self.stage1 = nn.Conv2d(32, 64, 3, padding=1)
            self.stage2 = nn.Conv2d(64, state_dim, 3, padding=1)
            self.stage3 = nn.GRU(state_dim, state_dim, batch_first=True)
            self.text_proj = nn.Linear(state_dim, text_dim)
            for module in (self.stage0, self.stage1):
                for p in module.parameters():
                    p.requires_grad = False

    def _forward_swin(self, frames: Tensor) -> DynFlowBackboneOutput:
        b, t, c, h, w = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).float()
        x = self.swin.patch_embed(x)
        x = self.swin.pos_drop(x)
        x = self.swin.features(x)
        x = self.swin.norm(x)  # B, _T, _H, _W, C
        x = self.state_proj(x)
        x_ch = x.permute(0, 4, 1, 2, 3).contiguous()
        local_ch = F.interpolate(x_ch, size=(t, 7, 7), mode="trilinear", align_corners=False)
        coarse_ch = F.adaptive_avg_pool3d(local_ch, (t, 4, 4))
        local = local_ch.permute(0, 2, 3, 4, 1).contiguous()
        coarse = coarse_ch.permute(0, 2, 3, 4, 1).contiguous()
        global_seq = local.mean(dim=(2, 3))
        text_tokens = self.text_proj(global_seq)
        return DynFlowBackboneOutput(local_grid=local, coarse_grid=coarse, global_sequence=global_seq, text_visual_tokens=text_tokens, forward_count=self.forward_count)

    def _forward_fallback(self, frames: Tensor) -> DynFlowBackboneOutput:
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w).float()
        x = F.relu(self.stage0(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.stage1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.stage2(x))
        local = F.adaptive_avg_pool2d(x, (7, 7)).permute(0, 2, 3, 1).reshape(b, t, 7, 7, -1)
        coarse = F.adaptive_avg_pool2d(x, (4, 4)).permute(0, 2, 3, 1).reshape(b, t, 4, 4, -1)
        pooled = x.mean(dim=(2, 3)).reshape(b, t, -1)
        global_seq, _ = self.stage3(pooled)
        text_tokens = self.text_proj(global_seq)
        return DynFlowBackboneOutput(local_grid=local, coarse_grid=coarse, global_sequence=global_seq, text_visual_tokens=text_tokens, forward_count=self.forward_count)

    def forward(self, frames: Tensor) -> DynFlowBackboneOutput:
        if frames.ndim != 5 or frames.shape[1] != 32 or frames.shape[2] != 3:
            raise ValueError(f"frames must be [B,32,3,H,W], got {tuple(frames.shape)}")
        self.forward_count += 1
        if self._uses_torchvision_swin:
            return self._forward_swin(frames)
        return self._forward_fallback(frames)
