from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PANEL_NAMES = (
    "Predicate Evidence Tubes",
    "Semantic Token Consolidation",
    "Corridor Flow Ribbons",
    "Pattern/State Lattice",
    "Event-Response Lag Ribbon",
    "Exact Signed Decision Waterfall",
    "Contribution-Aligned Text",
    "Counterfactual Twin",
)


def _panel_value(case: dict[str, Any], panel: str) -> Any:
    key = panel.lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    aliases = {
        "predicate_evidence_tubes": ("predicates", "predicate_tubes", "evidence_tubes"),
        "semantic_token_consolidation": ("semantic_tokens", "consolidation"),
        "corridor_flow_ribbons": ("corridor_flow", "traffic_flow"),
        "pattern_state_lattice": ("traffic_state", "pattern_state"),
        "event_response_lag_ribbon": ("response_lag", "lag"),
        "exact_signed_decision_waterfall": ("decision_ledger", "ledger"),
        "contribution_aligned_text": ("generated_text", "text_alignment"),
        "counterfactual_twin": ("interventions", "counterfactuals"),
    }
    for alias in aliases.get(key, (key,)):
        if alias in case:
            return case[alias]
    return {"missing": True, "required_panel": panel}


def render_case_canvas(case: dict, output_png: str | Path, output_json: str | Path) -> dict:
    output_png = Path(output_png)
    output_json = Path(output_json)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    panels = [{"name": name, "value": _panel_value(case, name)} for name in PANEL_NAMES]
    payload = {
        "schema": "acpr_dynflow_swin_canvas_v1",
        "sample_id": case.get("sample_id"),
        "git_sha": case.get("git_sha"),
        "config_hash": case.get("config_hash"),
        "panels": panels,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    from PIL import Image, ImageDraw

    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), f"ACPR-DynFlow-Swin Dynamic Traffic Decision Ledger | sample={case.get('sample_id')}", fill="black")
    panel_w, panel_h = 760, 210
    for idx, panel in enumerate(panels):
        col = idx % 2
        row = idx // 2
        x0 = 24 + col * (panel_w + 32)
        y0 = 60 + row * (panel_h + 24)
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        draw.rectangle((x0, y0, x1, y1), outline="black", width=2)
        draw.text((x0 + 12, y0 + 10), panel["name"], fill="black")
        value = panel["value"]
        if isinstance(value, dict) and value.get("missing"):
            text = "missing required tensor-linked record"
            color = "red"
        else:
            text = json.dumps(value, ensure_ascii=False)[:420]
            color = "black"
        draw.text((x0 + 12, y0 + 42), text, fill=color)
    image.save(output_png)
    return payload
