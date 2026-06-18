from __future__ import annotations

import json
from pathlib import Path

import torch


class FlowTraceRenderer:
    def render_canvas(self, bundle, output_dir: str | Path, sample_id: str = "sample") -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = bundle.to_diagnostics()
        meta["sample_id"] = sample_id
        json_path = out / f"{sample_id}_flowtrace_canvas.json"
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            from PIL import Image
            arr = bundle.state_evidence_maps[0, 0, 0].detach().float().cpu()
            arr = (255 * (arr - arr.min()) / (arr.max() - arr.min()).clamp_min(1e-6)).byte().numpy()
            Image.fromarray(arr).save(out / f"{sample_id}_flowtrace_canvas.png")
            meta["png"] = str(out / f"{sample_id}_flowtrace_canvas.png")
        except Exception as exc:
            meta["png_unavailable_reason"] = str(exc)
        return meta
