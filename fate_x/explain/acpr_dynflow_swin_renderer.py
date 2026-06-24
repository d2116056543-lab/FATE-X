from __future__ import annotations

import json
from pathlib import Path
import base64


def render_case_canvas(case: dict, output_png: str | Path, output_json: str | Path) -> dict:
    output_png = Path(output_png)
    output_json = Path(output_json)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "acpr_dynflow_swin_canvas_v1", "case": case}
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (640, 360), "white")
        draw = ImageDraw.Draw(image)
        draw.text((16, 16), "ACPR-DynFlow-Swin Canvas", fill="black")
        draw.text((16, 48), json.dumps(case, ensure_ascii=False)[:180], fill="black")
        image.save(output_png)
    except Exception:
        # Valid transparent 1x1 PNG fallback; still paired with source JSON.
        output_png.write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
            )
        )
    return payload
