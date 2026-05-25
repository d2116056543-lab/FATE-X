from __future__ import annotations

from argparse import Namespace

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
