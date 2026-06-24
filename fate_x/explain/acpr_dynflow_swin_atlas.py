from __future__ import annotations

import json
from pathlib import Path


def build_atlas(records: list[dict], output_html: str | Path, output_json: str | Path) -> dict:
    if not records:
        raise ValueError("atlas requires real records")
    tensor_sources = sorted(
        {
            panel["tensor_source"]
            for record in records
            for panel in record.get("panels", [])
            if panel.get("tensor_source")
        }
    )
    if not tensor_sources:
        raise ValueError("atlas records contain no tensor-linked panels")
    payload = {
        "schema": "acpr_dynflow_swin_atlas_v1",
        "case_count": len(records),
        "tensor_sources": tensor_sources,
        "records": records,
    }
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for record in records:
        panels = "".join(
            f"<li><code>{panel['tensor_source']}</code> shape={panel['shape']}</li>"
            for panel in record.get("panels", [])
        )
        rows.append(f"<section><h2>{record['sample_id']}</h2><ul>{panels}</ul></section>")
    Path(output_html).write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>ACPR-DynFlow-Swin Atlas</title>"
        "<style>body{font-family:Arial;margin:2rem}section{border:1px solid #bbb;padding:1rem;margin:1rem 0}</style>"
        f"</head><body><h1>ACPR-DynFlow-Swin Tensor Atlas</h1>{''.join(rows)}</body></html>",
        encoding="utf-8",
    )
    return payload
