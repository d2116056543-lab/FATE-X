import torch
import inspect

from fate_x.acpr_flow_v2.adapt_video_backbone import ADAPTVideoBackboneV2


class _FakeSwin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.backbone = type("Backbone", (), {"norm": type("Norm", (), {"normalized_shape": (1024,)})()})()

    def forward(self, x, return_stages=False):
        self.calls += 1
        b, _, t, _, _ = x.shape
        fine = torch.randn(b, 1024, t, 7, 7, device=x.device)
        coarse = torch.randn(b, 1024, max(1, t // 2), 4, 4, device=x.device)
        final = fine
        return (final, fine, coarse) if return_stages else final


def test_video_backbone_single_forward_and_dense_tokens():
    fake = _FakeSwin()
    model = ADAPTVideoBackboneV2(output_dim=8, adapt_feature_dim=512, video_swin=fake, latent_dim=1024)
    frames = torch.randn(1, 4, 3, 32, 32)
    out = model(frames)
    assert out.forward_count == 1
    assert fake.calls == 1
    assert out.fused_grid.shape[:2] == (1, 4)
    assert out.fused_grid.shape[-1] == 8
    assert out.dense_tokens_projected.shape[-1] == 512


def test_video_backbone_source_does_not_use_pooling_placeholder():
    source = inspect.getsource(ADAPTVideoBackboneV2)
    assert "adaptive_avg_pool" not in source
