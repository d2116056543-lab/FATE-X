from fate_x.explain.acpr_flowcal_v2_atlas import build_dataset_atlas


def test_atlas_schema_contains_index_and_html(tmp_path):
    out = build_dataset_atlas([{"sample_id": "s1", "traffic_state": "queue"}], tmp_path)
    assert out["index_path"].endswith(".json")
    assert out["html_path"].endswith(".html")
