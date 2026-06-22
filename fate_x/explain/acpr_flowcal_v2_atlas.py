from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path


def build_flowcal_v2_atlas(records: list[dict]) -> dict:
    return {"count": len(records), "records": records}


def build_dataset_atlas(records: list[dict], output_dir: str | Path) -> dict:
    """Create the V2 dataset-level atlas index and standalone HTML shell."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("traffic_state", "unknown"))].append(record)

    index = {
        "count": len(records),
        "groups": {name: len(rows) for name, rows in grouped.items()},
        "records": records,
    }
    index_path = output / "acpr_flowcal_v2_atlas_index.json"
    html_path = output / "acpr_flowcal_v2_atlas.html"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    rows = "\n".join(
        f"<li>{html.escape(str(name))}: {count}</li>" for name, count in sorted(index["groups"].items())
    )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>ACPR FlowCal V2 Atlas</title></head>"
        f"<body><h1>ACPR FlowCal V2 Atlas</h1><p>records: {len(records)}</p><ul>{rows}</ul></body></html>",
        encoding="utf-8",
    )
    return {"index_path": str(index_path), "html_path": str(html_path), "count": len(records)}
