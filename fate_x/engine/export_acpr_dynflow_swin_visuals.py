from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate_x.explain.acpr_dynflow_swin_renderer import render_case_canvas


def _load_cases(path: str | None) -> list[dict]:
    if not path:
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def export_visuals(records_jsonl: str | None, output_dir: str, max_cases: int = 8) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(records_jsonl)[:max_cases]
    if not cases:
        raise ValueError("visual export requires real fixed-case records JSONL; no demo/template export is allowed")
    index = []
    for idx, case in enumerate(cases):
        sample_id = str(case.get("sample_id", f"case_{idx}")).replace("/", "_").replace("\\", "_")
        png = out / f"{idx:04d}_{sample_id}.png"
        src = out / f"{idx:04d}_{sample_id}.json"
        payload = render_case_canvas(case, png, src)
        index.append({"sample_id": sample_id, "png": str(png), "json": str(src), "schema": payload.get("schema")})
    summary = {"schema": "acpr_dynflow_swin_visual_index_v1", "case_count": len(index), "cases": index}
    (out / "visual_artifact_index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_cases", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(export_visuals(args.records_jsonl, args.output_dir, args.max_cases), indent=2))


if __name__ == "__main__":
    main()
