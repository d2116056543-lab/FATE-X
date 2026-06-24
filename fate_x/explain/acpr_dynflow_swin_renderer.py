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


def _tensor_record(source: str, tensor, max_values: int = 64) -> dict[str, Any]:
    detached = tensor.detach().float().cpu()
    return {
        "tensor_source": source,
        "shape": list(detached.shape),
        "values": detached.flatten()[:max_values].tolist(),
        "finite": bool(detached.isfinite().all()),
        "min": float(detached.min()) if detached.numel() else 0.0,
        "max": float(detached.max()) if detached.numel() else 0.0,
        "mean": float(detached.mean()) if detached.numel() else 0.0,
    }


def tensor_case_from_output(
    output,
    sample_id: str,
    git_sha: str,
    config_hash: str,
    checkpoint_hash: str,
    counterfactuals: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "checkpoint_hash": checkpoint_hash,
        "predicates": _tensor_record("predicates.evidence_maps", output.predicates.evidence_maps),
        "semantic_tokens": _tensor_record("semantic_tokens.assignment", output.semantic_tokens.assignment),
        "corridor_flow": _tensor_record("predicates.corridor_mass", output.predicates.corridor_mass),
        "traffic_state": _tensor_record("traffic.factor_probs", output.traffic.factor_probs),
        "response_lag": _tensor_record("traffic.lag_weights", output.traffic.lag_weights),
        "decision_ledger": _tensor_record(
            "ledger.gated_factor_contributions_normalized",
            output.ledger.gated_factor_contributions_normalized,
        ),
        "generated_text": _tensor_record(
            "text.explanation_to_factor_attention",
            output.text.explanation_to_factor_attention,
        ),
        "interventions": {
            "tensor_source": "intervention_audit.output_deltas",
            "shape": [len(counterfactuals)],
            "values": [float(value) for value in counterfactuals.values()],
            "names": list(counterfactuals),
        },
    }


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
    panels = []
    for name in PANEL_NAMES:
        value = _panel_value(case, name)
        if not isinstance(value, dict) or value.get("missing"):
            raise ValueError(f"{name} requires a real tensor-linked record")
        for required in ("tensor_source", "shape", "values"):
            if required not in value:
                raise ValueError(f"{name} missing {required}")
        panels.append(
            {
                "name": name,
                "tensor_source": str(value["tensor_source"]),
                "shape": list(value["shape"]),
                "values": value["values"],
                "metadata": {k: v for k, v in value.items() if k not in {"tensor_source", "shape", "values"}},
            }
        )
    for required in ("sample_id", "git_sha", "config_hash", "checkpoint_hash"):
        if not case.get(required):
            raise ValueError(f"canvas source record missing {required}")
    payload = {
        "schema": "acpr_dynflow_swin_canvas_v1",
        "sample_id": case.get("sample_id"),
        "git_sha": case.get("git_sha"),
        "config_hash": case.get("config_hash"),
        "checkpoint_hash": case.get("checkpoint_hash"),
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
        values = panel["values"]
        flat = values if isinstance(values, list) else [values]
        flat = [float(value) for value in flat[:64] if isinstance(value, (int, float))]
        draw.text(
            (x0 + 12, y0 + 38),
            f"{panel['tensor_source']} shape={panel['shape']}",
            fill="black",
        )
        if flat:
            minimum, maximum = min(flat), max(flat)
            span = max(maximum - minimum, 1e-6)
            chart_y0, chart_y1 = y0 + 75, y1 - 18
            chart_x0, chart_x1 = x0 + 12, x1 - 12
            points = []
            for point_index, value in enumerate(flat):
                x = chart_x0 + point_index * (chart_x1 - chart_x0) / max(len(flat) - 1, 1)
                y = chart_y1 - (value - minimum) / span * (chart_y1 - chart_y0)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill="navy", width=3)
            else:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="navy")
    image.save(output_png)
    return payload
