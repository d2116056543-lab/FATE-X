import pytest
import torch

from fate_x.models.dynamic_traffic_state_composer import DynamicTrafficStateComposer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16 kernel coverage")
def test_dynamic_traffic_state_composer_accepts_bfloat16_cuda():
    model = DynamicTrafficStateComposer(state_dim=256, num_states=8, num_tracks=12, heads=4).cuda().to(dtype=torch.bfloat16)
    track_tokens = torch.randn(1, 32, 12, 256, device="cuda", dtype=torch.bfloat16)
    track_attention = torch.randn(1, 32, 12, 7, 7, device="cuda", dtype=torch.bfloat16).softmax(dim=-1)
    relative_motion = torch.randn(1, 31, 12, 2, device="cuda", dtype=torch.bfloat16)
    unmatched = torch.rand(1, 32, 12, device="cuda", dtype=torch.bfloat16)

    out = model(track_tokens, track_attention, relative_motion, unmatched)

    assert out["state_tokens_temporal"].shape == (1, 32, 8, 256)
    assert out["state_memory"].shape == (1, 8, 256)
    assert torch.isfinite(out["state_tokens_temporal"]).all()
    assert torch.isfinite(out["state_evidence_maps"]).all()
