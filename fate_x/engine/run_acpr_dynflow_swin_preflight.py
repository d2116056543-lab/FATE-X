from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_x.acpr_dynflow_swin.config import load_config, build_config_consumer_manifest
from fate_x.engine.train_acpr_dynflow_swin import smoke_batch
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel


def run_preflight(config: str, output_dir: str) -> dict:
    cfg = load_config(config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = ACPRDynFlowSwinModel(cfg)
    batch = smoke_batch()
    output = model(batch)
    reports = {
        "config_binding_report.json": build_config_consumer_manifest(cfg),
        "tensor_contracts.json": {
            "total_loss_finite": bool(torch.isfinite(output.total_loss).item()),
            "backbone_forward_count": output.backbone.forward_count,
            "predicate_count": len(output.predicates.names),
            "traffic_factor_count": len(output.traffic.factor_names),
        },
        "review_report.json": {"passed": False, "reason": "review pass requires full dynamic gates, not smoke only"},
    }
    for name, report in reports.items():
        (out_dir / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return reports["tensor_contracts.json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--output_dir", default=".background_runs/acpr_dynflow_swin_v1_preflight")
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
