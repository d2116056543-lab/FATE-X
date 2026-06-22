from pathlib import Path


def test_v2_formal_import_graph_rejects_legacy_namespace_strings():
    root = Path.cwd()
    v2_files = list((root / "fate_x/acpr_flow_v2").glob("*.py"))
    forbidden = [
        "fate_x.acpr_flow.model",
        "fate_x.models.token_pmt_adapter",
        "fate_x.models.sinkhorn_transport",
        "fate_x.losses.flowtrace_losses",
        "fate_x.explain.flowtrace_renderer",
    ]
    hits = []
    for path in v2_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits.extend((path.name, needle) for needle in forbidden if needle in text)
    assert not hits
