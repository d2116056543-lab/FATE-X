from __future__ import annotations

import argparse
import json
import time

import torch

from fate_x.acpr_dynflow_swin.config import load_config
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.engine.acpr_dynflow_swin_data import build_dynflow_swin_dataloader
from fate_x.engine.train_acpr_dynflow_swin import move_batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ACPRDynFlowSwinModel(cfg).to(device)
    loader = build_dynflow_swin_dataloader(
        cfg,
        split="train",
        batch_size=args.batch_size,
        max_samples=max(args.max_samples, args.batch_size * (args.steps + args.warmup_steps)),
        synthetic=args.synthetic,
    )
    iterator = iter(loader)
    for _ in range(args.warmup_steps):
        batch = move_batch_to_device(next(iterator), device)
        out = model(batch)
        out.total_loss.backward()
        model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    measured = 0
    for _ in range(args.steps):
        batch = move_batch_to_device(next(iterator), device)
        out = model(batch)
        out.total_loss.backward()
        model.zero_grad(set_to_none=True)
        measured += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    samples_per_second = measured * args.batch_size / max(elapsed, 1e-6)
    train_count = len(getattr(loader, "dataset", [])) if getattr(loader, "dataset", None) is not None else None
    projected_hours = None
    if train_count:
        projected_hours = train_count / max(samples_per_second, 1e-6) / 3600.0
    gate_hours = float(cfg.get("memory_throughput_probe", {}).get("projected_train_epoch_limit_hours", 4.0))
    result = {
        "measured_steps": measured,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples_per_second,
        "projected_epoch_hours": projected_hours,
        "projected_epoch_limit_hours": gate_hours,
        "formal_gate_passed": projected_hours is not None and projected_hours <= gate_hours,
        "synthetic": bool(args.synthetic),
        "device": str(device),
    }
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
