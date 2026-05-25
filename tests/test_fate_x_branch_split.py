from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from fate_x.engine.fate_x_compat import validate_fate_x_mask_compatibility
from fate_x.engine.smoke_fate_x_forward import run_smoke


def test_text_branch_reduces_tokens_control_branch_stays_dense():
    args = Namespace(
        output="unused.json",
        video_token_reducer="topk_merge",
        temporal_evidence_memory="queries",
        batch_size=1,
        num_tokens=32,
        max_img_seq_length=32,
        text_len=5,
        dim=16,
        keep_ratio=0.5,
        num_summary_tokens=2,
        min_tokens=4,
        num_events=2,
        seed=3,
        fate_x_text_reduce_only=True,
        fate_x_reduce_control=False,
    )
    result = run_smoke(args)
    assert result["dense_visual_tokens"] == 32
    assert result["text_visual_tokens"] < 32
    assert result["control_visual_tokens"] == 32
    assert result["control_branch_dense"] is True
    assert result["fate_x_text_reduce_only"] is True
    assert result["fate_x_reduce_control"] is False


def test_reduce_control_requires_temporal_order_preserving_reducer():
    args = Namespace(
        fate_x_enabled=True,
        video_token_reducer="topk_merge",
        fate_x_reduce_control=True,
        fate_x_control_reducer="none",
        learn_mask_enabled=False,
    )
    with pytest.raises(ValueError):
        validate_fate_x_mask_compatibility(args)


def test_imported_multitask_model_contains_text_control_branch_split():
    source = Path("src/modeling/multitask_e2e_vid_swin_bert.py").read_text(encoding="utf-8")
    run_adapt = Path("src/tasks/run_adapt.py").read_text(encoding="utf-8")
    assert "from src.modeling.multitask_e2e_vid_swin_bert import MultitaskVideoTransformer" in run_adapt
    assert "vid_feats_dense" in source
    assert "vid_feats_text" in source
    assert "vid_feats_control" in source
    assert "control_kwargs" in source
    assert "self.sensor_pred_head(*args, **control_kwargs)" in source
    assert "self.trans_encoder(*args, **text_kwargs)" in source


def test_reduce_control_per_frame_allowed_when_not_text_only():
    args = Namespace(
        fate_x_enabled=True,
        video_token_reducer="per_frame_topk_merge",
        fate_x_reduce_control=True,
        fate_x_text_reduce_only=False,
        fate_x_control_reducer="per_frame_topk_merge",
        learn_mask_enabled=False,
    )
    validate_fate_x_mask_compatibility(args)


def test_text_only_and_reduce_control_are_inconsistent():
    args = Namespace(
        fate_x_enabled=True,
        video_token_reducer="per_frame_topk_merge",
        fate_x_reduce_control=True,
        fate_x_text_reduce_only=True,
        fate_x_control_reducer="per_frame_topk_merge",
        learn_mask_enabled=False,
    )
    with pytest.raises(ValueError, match="fate_x_text_reduce_only"):
        validate_fate_x_mask_compatibility(args)


def test_learn_mask_rejected_with_token_compression():
    args = Namespace(
        fate_x_enabled=True,
        video_token_reducer="topk_merge",
        fate_x_reduce_control=False,
        fate_x_text_reduce_only=True,
        fate_x_control_reducer="none",
        learn_mask_enabled=True,
    )
    with pytest.raises(ValueError, match="learn_mask_enabled"):
        validate_fate_x_mask_compatibility(args)
