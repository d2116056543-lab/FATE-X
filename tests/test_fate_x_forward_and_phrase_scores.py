from __future__ import annotations

import torch

from fate_x.engine.generate_phrase_scores import generate_phrase_score_rows
from fate_x.engine.smoke_fate_x_forward import run_smoke
from argparse import Namespace


def test_smoke_fate_x_forward_resizes_attention_mask():
    args = Namespace(
        output="unused.json",
        video_token_reducer="topk_merge",
        temporal_evidence_memory="queries",
        batch_size=2,
        num_tokens=32,
        max_img_seq_length=32,
        text_len=5,
        dim=16,
        keep_ratio=0.5,
        num_summary_tokens=4,
        min_tokens=4,
        num_events=3,
        seed=1,
    )
    result = run_smoke(args)
    assert result["output_shape"][1] < 32 + 3
    assert result["attention_mask_shape"][-1] == 5 + result["output_shape"][1]
    assert result["has_provenance"] is True


def test_generate_phrase_score_rows_from_token_scores():
    rows = [{"prediction": "A pedestrian is crossing in front of the car."}]
    token_scores = torch.linspace(0, 1, steps=10).view(1, 10)
    out, summary = generate_phrase_score_rows(rows, token_scores=token_scores, topk_ratio=0.2)
    assert out[0]["phrase_hit_count"] >= 1
    assert "phrase_faithfulness" in out[0]
    assert summary["faithfulness_available"] is True
    assert summary["with_generated_scores"] == 1