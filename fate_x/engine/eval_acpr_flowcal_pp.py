from __future__ import annotations

import argparse

import torch

from fate_x.acpr_flow.model import ACPRFlowModel
from fate_x.utils.acpr_flow_artifacts import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = ACPRFlowModel().to(args.device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device)["model"], strict=False)
    with torch.no_grad():
        out = model(torch.randn(1, 32, 3, 224, 224, device=args.device))
    write_json(f"{args.output_dir}/eval_report.json", {"total_loss": float(out.total_loss.cpu()), "reason_tokens": out.bundle.reason_memory.shape[1]})


if __name__ == "__main__":
    main()
