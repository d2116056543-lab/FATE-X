from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_x.acpr_dynflow.config import load_dynflow_config
from fate_x.acpr_dynflow.model import ACPRDynFlowModel
from fate_x.engine.acpr_dynflow_data import build_dynflow_dataloader


def evaluate(config: str, checkpoint: str | None, output_dir: str, device: str = "cpu", max_samples: int = 8, synthetic: bool = False) -> dict:
    cfg = load_dynflow_config(config)
    loader = build_dynflow_dataloader(cfg.raw, "test", batch_size=1, max_samples=max_samples, synthetic=synthetic)
    model = ACPRDynFlowModel(cfg).to(device)
    if checkpoint and Path(checkpoint).exists():
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(data.get("model", data), strict=False)
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.frames = batch.frames.to(device)
            if batch.input_ids is not None:
                batch.input_ids = batch.input_ids.to(device)
            if batch.masked_ids is not None:
                batch.masked_ids = batch.masked_ids.to(device)
            if batch.control_target is not None:
                batch.control_target = batch.control_target.to(device)
            out = model(batch)
            preds.append(out.ledger.final_prediction_raw.cpu())
            if batch.control_target is not None:
                targets.append(batch.control_target.cpu())
    metrics = {"control_metrics_available": False}
    if preds and targets:
        metrics = model.codec.official_metrics(torch.cat(preds), torch.cat(targets))
        metrics["control_metrics_available"] = True
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_samples", type=int, default=8)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()

