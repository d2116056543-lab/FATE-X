from __future__ import annotations

from pathlib import Path

import torch

from fate_x.acpr_flow.types import ACPRFlowBundle
from fate_x.utils.acpr_flow_artifacts import write_json


def render_acpr_flow_canvas(bundle: ACPRFlowBundle, output_dir: str | Path, sample_id: str = "sample") -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_id": sample_id,
        "predicate_names": bundle.diagnostics.get("predicate_names", []),
        "flow_factor_names": bundle.diagnostics.get("flow_factor_names", []),
        "predicate_prob_mean": bundle.predicate_probs_temporal.detach().float().mean(dim=(0, 1)).cpu().tolist(),
        "flow_prob_mean": bundle.flow_probs.detach().float().mean(dim=0).cpu().tolist(),
        "control_delta_norm": float(bundle.control_delta.detach().float().norm().cpu()) if bundle.control_delta is not None else 0.0,
    }
    json_path = output_dir / f"{sample_id}_acpr_flow_canvas.json"
    write_json(json_path, payload)
    png_path = output_dir / f"{sample_id}_acpr_flow_canvas.ppm"
    grid = bundle.predicate_attention[0, -1].detach().float().mean(0).cpu()
    grid = (255 * (grid - grid.min()) / (grid.max() - grid.min()).clamp_min(1e-6)).to(torch.uint8)
    h, w = grid.shape
    data = b"P5\n%d %d\n255\n" % (w, h) + grid.numpy().tobytes()
    png_path.write_bytes(data)
    return {"json": str(json_path), "image": str(png_path)}
