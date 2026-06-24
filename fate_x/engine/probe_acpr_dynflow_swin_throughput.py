from __future__ import annotations

import argparse
import json
import time

from fate_x.acpr_dynflow_swin.config import load_config
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.engine.train_acpr_dynflow_swin import smoke_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = ACPRDynFlowSwinModel(cfg)
    start = time.perf_counter()
    for _ in range(args.steps):
        out = model(smoke_batch())
        out.total_loss.backward()
        model.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - start
    print(json.dumps({"measured_steps": args.steps, "elapsed_seconds": elapsed, "samples_per_second": args.steps / max(elapsed, 1e-6), "formal_gate_passed": False}, indent=2))


if __name__ == "__main__":
    main()
