import torch
from fate_x.models.flowtrace_pmt_model import FlowTracePMTModel
from fate_x.explain.flowtrace_renderer import FlowTraceRenderer


def test_renderer_writes_json(tmp_path):
    model = FlowTracePMTModel(fine_dim=8, coarse_dim=16, dense_dim=16, state_dim=8, num_tracks=2, num_states=2)
    b = model(torch.randn(1, 4, 16), torch.randn(1, 8, 3, 4, 4), torch.randn(1, 16, 3, 2, 2))
    meta = FlowTraceRenderer().render_canvas(b, tmp_path, "x")
    assert (tmp_path / "x_flowtrace_canvas.json").exists()
    assert meta["state_evidence_maps_shape"] == list(b.state_evidence_maps.shape)
