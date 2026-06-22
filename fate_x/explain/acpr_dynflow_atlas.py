from __future__ import annotations

from pathlib import Path


def build_atlas(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "atlas.html"
    path.write_text("<html><body>ACPR-DynFlow atlas</body></html>", encoding="utf-8")
    return path

