from __future__ import annotations

from argparse import Namespace

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
