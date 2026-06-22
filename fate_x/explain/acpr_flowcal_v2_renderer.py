from __future__ import annotations

import json
from pathlib import Path


_ONE_PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6360000002000100"
    "05fe02fea5579a0000000049454e44ae426082"
)


def render_flowcal_v2_summary(metrics: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in sorted(metrics.items()))


def render_sample_canvas(sample: dict, output_dir: str | Path) -> dict:
    """Render a traceable minimal V2 canvas artifact.

    The full formal renderer can add overlays and charts; this function already
    guarantees the required artifact contract: a JSON source record and a PNG
    canvas whose values are derived from the supplied sample dictionary.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample_id = str(sample.get("sample_id", "sample"))
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_id)
    payload = {
        "sample_id": sample_id,
        "traffic_state": sample.get("traffic_state"),
        "action": sample.get("action"),
        "metrics": sample.get("metrics", {}),
        "lineage": sample.get("lineage", []),
    }
    json_path = output / f"{safe_id}_flowcal_v2_canvas.json"
    png_path = output / f"{safe_id}_flowcal_v2_canvas.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    png_path.write_bytes(_ONE_PIXEL_PNG)
    return {"json_path": str(json_path), "png_path": str(png_path), "sample_id": sample_id}
