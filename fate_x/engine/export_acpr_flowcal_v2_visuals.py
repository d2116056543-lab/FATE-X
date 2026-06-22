from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_visuals(output_dir: str, sample_id: str = "sample") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"sample_id": sample_id, "panels": ["predicate", "lane_flow", "reason_graph", "control"]}
    (out / f"{sample_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sample_id", default="sample")
    args = p.parse_args()
    print(json.dumps(export_visuals(args.output_dir, args.sample_id), indent=2))


if __name__ == "__main__":
    main()
