from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate_x.explain.acpr_dynflow_swin_atlas import build_atlas


def _load_records(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("cases") or data.get("records") or []
    rows = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="visual_artifact_index.json or records.jsonl")
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    records = _load_records(args.records)
    if not records:
        raise SystemExit("Atlas requires non-empty real visual records")
    print(json.dumps(build_atlas(records, args.output_html, args.output_json), indent=2))


if __name__ == "__main__":
    main()
