from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_atlas(output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {"groups": ["clear_open_flow", "queue_congestion"], "items": []}
    (out / "atlas_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (out / "atlas.html").write_text("<html><body><h1>ACPR FlowCal V2 Atlas</h1></body></html>\n", encoding="utf-8")
    return index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    print(json.dumps(build_atlas(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
