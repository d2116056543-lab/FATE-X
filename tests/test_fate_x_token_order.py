from __future__ import annotations

import torch

from fate_x.models.video_token_reducer import VideoTokenReducer


def test_topk_kept_tokens_preserve_original_order():
    tokens = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]])
    reducer = VideoTokenReducer(1, keep_ratio=0.5, num_summary_tokens=0, min_tokens=1, mode="topk_merge")
    out = reducer(tokens, attention_proxy=scores)
    kept = out["tokens"][0, :, 0].tolist()
    assert kept == sorted(kept)


def test_per_frame_topk_preserves_frame_order():
    tokens = torch.arange(12, dtype=torch.float32).view(1, 12, 1)
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0, 0.95, 0.15]])
    reducer = VideoTokenReducer(
        1,
        keep_ratio=0.5,
        num_summary_tokens=0,
        min_tokens=1,
        mode="per_frame_topk_merge",
        temporal_tokens=3,
        spatial_tokens_per_frame=4,
        min_tokens_per_frame=1,
    )
    out = reducer(tokens, attention_proxy=scores)
    kept = out["tokens"][0, :, 0].tolist()
    assert kept == sorted(kept)
    assert len(kept) >= 3
