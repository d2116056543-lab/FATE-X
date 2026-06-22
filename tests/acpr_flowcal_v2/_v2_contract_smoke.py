from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch


def repo_root() -> Path:
    return Path.cwd()


def assert_symbol(module_name: str, symbol: str):
    module = __import__(module_name, fromlist=[symbol])
    assert hasattr(module, symbol), f"{module_name}.{symbol} missing"
    return getattr(module, symbol)


def assert_manifest_symbol(module_name: str, symbol: str):
    obj = assert_symbol(module_name, symbol)
    assert obj is not None
    return obj


def make_reason_memory(batch: int = 1, dim: int = 8):
    from fate_x.acpr_flow_v2.types import SemanticReasonMemory
    return SemanticReasonMemory(
        values=torch.randn(batch, 54, dim),
        mask=torch.ones(batch, 54, dtype=torch.bool),
        confidence=torch.ones(batch, 54),
        names=tuple(f"m{i}" for i in range(54)),
        type_ids=torch.arange(54) % 6,
        axis_ids=torch.tensor([3] * 48 + [1, 2, 3, 3, 3, 0], dtype=torch.long),
        evidence_maps=torch.zeros(batch, 2, 54, 2, 2),
        lineage=[{"source": f"m{i}"} for i in range(54)],
        semantic_state=torch.randn(batch, dim),
    )


def run_tiny_forward():
    from fate_x.acpr_flow_v2.config import ACPRFlowCalV2Config
    from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
    from fate_x.acpr_flow_v2.types import FlowCalV2Batch

    cfg = ACPRFlowCalV2Config(hidden_dim=16, text_vocab_size=13, num_frames=32)
    model = ACPRFlowCalV2Model(cfg)
    batch = FlowCalV2Batch(
        frames=torch.randn(1, 32, 3, 64, 64),
        input_ids=torch.randint(0, 13, (1, 30)),
        attention_mask=torch.ones(1, 30, dtype=torch.long),
        masked_pos=torch.tensor([[0, 1]]),
        masked_ids=torch.randint(0, 13, (1, 2)),
        car_info=torch.randn(1, 2, 32),
        sample_ids=["tiny"],
        raw_actions=["slow"],
        raw_justifications=["traffic ahead"],
    )
    out = model(batch, stage="R")
    assert torch.isfinite(out.total_loss)
    return out
