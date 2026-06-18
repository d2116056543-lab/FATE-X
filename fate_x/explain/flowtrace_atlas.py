from __future__ import annotations

import json
from pathlib import Path


class FlowTraceAtlasBuilder:
    def build(self, records: list[dict], output_dir: str | Path) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        atlas = {"num_records": len(records), "records": records[:100]}
        (out / "flowtrace_atlas.json").write_text(json.dumps(atlas, indent=2), encoding="utf-8")
        return atlas
