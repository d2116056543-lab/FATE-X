import torch

from fate_x.acpr_flow.model import ACPRFlowModel
from fate_x.explain.acpr_flow_renderer import render_acpr_flow_canvas


def test_visual_renderer_writes_named_json_and_image(tmp_path):
    model = ACPRFlowModel()
    with torch.no_grad():
        bundle = model(torch.randn(1, 32, 3, 64, 64)).bundle
    out = render_acpr_flow_canvas(bundle, tmp_path, "case")
    assert out["json"].endswith(".json")
    assert out["image"].endswith(".ppm")
