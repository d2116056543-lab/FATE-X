from fate_x.explain.acpr_flowcal_v2_renderer import render_sample_canvas


def test_renderer_returns_json_and_png_paths(tmp_path):
    row = render_sample_canvas({"sample_id": "s1", "traffic_state": "queue"}, tmp_path)
    assert row["json_path"].endswith(".json")
    assert row["png_path"].endswith(".png")
