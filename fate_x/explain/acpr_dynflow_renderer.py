from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_dynamic_traffic_decision_ledger(case: dict[str, Any], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ledger_case.json"
    path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return path

