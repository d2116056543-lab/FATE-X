from __future__ import annotations

import json
from pathlib import Path


def build_atlas(records: list[dict], output_html: str | Path, output_json: str | Path) -> dict:
    payload = {"schema": "acpr_dynflow_swin_atlas_v1", "records": records}
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(output_html).write_text("<html><body><pre>ACPR-DynFlow-Swin Atlas</pre></body></html>", encoding="utf-8")
    return payload
