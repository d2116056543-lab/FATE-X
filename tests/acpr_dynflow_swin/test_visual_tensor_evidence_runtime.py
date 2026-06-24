import json

import torch

from fate_x.acpr_dynflow_swin.types import (
    ExactDecisionLedger,
    SemanticTokenConsolidation,
    TrafficStateOutput,
)
from fate_x.explain.acpr_dynflow_swin_atlas import build_atlas
from fate_x.explain.acpr_dynflow_swin_renderer import render_case_canvas


def _case():
    return {
        "sample_id": "case-1",
        "git_sha": "abc",
        "config_hash": "def",
        "checkpoint_hash": "ghi",
        "predicates": {"tensor_source": "predicates.evidence_maps", "shape": [1, 2, 32, 7, 7], "values": [0.1, 0.2]},
        "semantic_tokens": {"tensor_source": "semantic_tokens.assignment", "shape": [1, 16, 49, 5], "values": [0.2] * 5},
        "corridor_flow": {"tensor_source": "predicates.corridor_mass", "shape": [1, 2, 32, 3], "values": [0.1, 0.5, 0.4]},
        "traffic_state": {"tensor_source": "traffic.factor_probs", "shape": [1, 2, 13], "values": [0.1] * 13},
        "response_lag": {"tensor_source": "traffic.lag_weights", "shape": [1, 32, 13, 4], "values": [0.1, 0.2, 0.3, 0.4]},
        "decision_ledger": {"tensor_source": "ledger.gated_factor_contributions_normalized", "shape": [1, 32, 13, 2], "values": [0.1, -0.1]},
        "generated_text": {"tensor_source": "text.explanation_to_factor_attention", "shape": [1, 13], "values": [0.1] * 13},
        "interventions": {"tensor_source": "intervention_audit", "shape": [2], "values": [0.0, 0.2]},
    }


def test_canvas_rejects_missing_tensor_linkage(tmp_path):
    case = _case()
    del case["predicates"]["tensor_source"]
    try:
        render_case_canvas(case, tmp_path / "x.png", tmp_path / "x.json")
    except ValueError as exc:
        assert "tensor_source" in str(exc)
    else:
        raise AssertionError("missing tensor linkage must block rendering")


def test_canvas_and_atlas_contain_real_evidence(tmp_path):
    case = _case()
    payload = render_case_canvas(case, tmp_path / "x.png", tmp_path / "x.json")
    atlas = build_atlas([payload], tmp_path / "atlas.html", tmp_path / "atlas.json")
    assert (tmp_path / "x.png").stat().st_size > 1000
    assert all(panel["tensor_source"] for panel in payload["panels"])
    assert atlas["case_count"] == 1
    assert atlas["tensor_sources"]
    assert "case-1" in (tmp_path / "atlas.html").read_text(encoding="utf-8")
