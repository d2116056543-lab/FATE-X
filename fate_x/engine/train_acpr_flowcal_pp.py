from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_x.acpr_flow.model import ACPRFlowModel
from fate_x.utils.acpr_flow_config import load_acpr_flow_config


def build_acpr_optimizer_groups(model: ACPRFlowModel) -> tuple[list[dict], dict[str, str]]:
    lr_by_prefix = {
        "predicate_field": 1e-4,
        "flow_composer": 1e-4,
        "reason_memory": 1e-4,
        "temporal_seca": 5e-5,
        "reason_control_adapter": 2e-5,
        "prefix_future_head": 5e-5,
        "backbone": 5e-6,
        "control_base": 1e-5,
        "control_hidden": 1e-5,
    }
    groups: dict[float, dict] = {}
    manifest = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = next((p for p in lr_by_prefix if name.startswith(p)), "new_modules")
        lr = lr_by_prefix.get(matched, 1e-4)
        groups.setdefault(lr, {"params": [], "lr": lr, "names": []})
        groups[lr]["params"].append(param)
        groups[lr]["names"].append(name)
        manifest[name] = matched
    return list(groups.values()), manifest


def train_smoke(config: str, output_dir: str, device: str = "cpu", max_steps: int = 8,
                batch_size: int = 1, epochs: int = 1) -> None:
    cfg = load_acpr_flow_config(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ACPRFlowModel().to(device)
    groups, manifest = build_acpr_optimizer_groups(model)
    opt = torch.optim.AdamW(groups)
    (out / "optimizer_group_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    status = []
    for epoch in range(epochs):
        for step in range(max_steps):
            frames = torch.randn(batch_size, cfg["data"]["max_num_frames"], 3, cfg["data"]["image_resolution"], cfg["data"]["image_resolution"], device=device)
            res = model(frames)
            opt.zero_grad(set_to_none=True)
            res.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rec = {"epoch": epoch, "global_step": epoch * max_steps + step + 1, "loss": float(res.total_loss.detach().cpu()), "frames_shape": list(frames.shape)}
            status.append(rec)
            print("ACPR_FLOW_BATCH " + json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch}, out / "checkpoint_latest.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch}, out / "checkpoint_best_test.pth")
    (out / "metrics_summary.jsonl").write_text("\n".join(json.dumps(x) for x in status), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=1)
    args = parser.parse_args()
    train_smoke(args.config, args.output_dir, args.device, args.max_steps, args.batch_size, args.epochs)


if __name__ == "__main__":
    main()
