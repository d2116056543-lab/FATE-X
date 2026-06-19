from __future__ import annotations

import argparse

import torch

from fate_x.acpr_flow.model import ACPRFlowModel
from fate_x.explain.acpr_flow_renderer import render_acpr_flow_canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = ACPRFlowModel().to(args.device)
    with torch.no_grad():
        bundle = model(torch.randn(1, 32, 3, 224, 224, device=args.device)).bundle
    print(render_acpr_flow_canvas(bundle, args.output_dir))


if __name__ == "__main__":
    main()
