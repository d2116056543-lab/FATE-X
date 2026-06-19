from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fate_x.utils.acpr_flow_artifacts import write_json


def build_acpr_flow_atlas(records: Iterable[dict], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    payload = {"count": len(records), "records": records}
    path = output_dir / "acpr_flow_atlas.json"
    write_json(path, payload)
    return {"atlas_json": str(path)}
