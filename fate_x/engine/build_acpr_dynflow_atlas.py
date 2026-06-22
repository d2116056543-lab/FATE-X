from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "atlas.html").write_text("<html><body><h1>ACPR-DynFlow Atlas</h1></body></html>", encoding="utf-8")
    (out / "atlas_index.json").write_text('{"atlas_html": true}', encoding="utf-8")


if __name__ == "__main__":
    main()

