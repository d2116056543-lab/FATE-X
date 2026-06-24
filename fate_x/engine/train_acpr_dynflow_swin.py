from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from fate_x.acpr_dynflow_swin.config import load_config
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.acpr_dynflow_swin.types import DynFlowSwinBatch
from fate_x.engine.acpr_dynflow_swin_data import build_dynflow_swin_dataloader


def build_optimizer_groups(model: torch.nn.Module, cfg: dict) -> list[dict]:
    lr_cfg = cfg.get("optimization", {}).get("learning_rates", {})
    groups = []
    seen: set[int] = set()
    routing = [
        ("backbone", float(lr_cfg.get("video_swin_backbone", 1e-5))),
        ("predicates", float(lr_cfg.get("predicate_query_temporal", 1e-4))),
        ("text", float(lr_cfg.get("caption_bert_and_lm_head", 5e-5))),
    ]
    for prefix, lr in routing:
        params = [p for name, p in model.named_parameters() if name.startswith(prefix) and p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "name": prefix})
            seen.update(id(p) for p in params)
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if rest:
        groups.append({"params": rest, "lr": float(lr_cfg.get("new_motion_traffic_ledger_heads", 2e-4)), "name": "new_motion_traffic_ledger_heads"})
    return groups


def smoke_batch(batch_size: int = 1, image_size: int = 32) -> DynFlowSwinBatch:
    return DynFlowSwinBatch(
        frames=torch.randn(batch_size, 32, 3, image_size, image_size),
        input_ids=torch.randint(0, 100, (batch_size, 30)),
        attention_mask=torch.ones(batch_size, 30, dtype=torch.long),
        token_type_ids=torch.zeros(batch_size, 30, dtype=torch.long),
        masked_pos=torch.ones(batch_size, 30, dtype=torch.long),
        masked_ids=torch.randint(0, 100, (batch_size, 30)),
        control_target=torch.randn(batch_size, 32, 2),
        sample_ids=[f"smoke-{idx}" for idx in range(batch_size)],
        raw_actions=["action"] * batch_size,
        raw_justifications=["reason"] * batch_size,
    )


def replace_link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        import shutil

        shutil.copy2(source, target)


def move_batch_to_device(batch: DynFlowSwinBatch, device: torch.device) -> DynFlowSwinBatch:
    batch.frames = batch.frames.to(device, non_blocking=True)
    batch.input_ids = batch.input_ids.to(device, non_blocking=True)
    batch.attention_mask = batch.attention_mask.to(device, non_blocking=True)
    batch.token_type_ids = batch.token_type_ids.to(device, non_blocking=True)
    batch.masked_pos = batch.masked_pos.to(device, non_blocking=True)
    batch.masked_ids = batch.masked_ids.to(device, non_blocking=True)
    batch.control_target = batch.control_target.to(device, non_blocking=True)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--output_dir", default=".background_runs/acpr_dynflow_swin_v1_train")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--smoke_steps", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ACPRDynFlowSwinModel(cfg).to(device)
    optimizer = torch.optim.AdamW(build_optimizer_groups(model, cfg))
    batch_size = args.batch_size or int(cfg.get("optimization", {}).get("batch_size", 1))
    if args.smoke_steps > 0 and args.max_steps is None:
        args.max_steps = args.smoke_steps
        args.synthetic = True
    loader = build_dynflow_swin_dataloader(
        cfg,
        split="train",
        batch_size=batch_size,
        max_samples=args.max_train_samples,
        synthetic=args.synthetic,
    )
    best_loss = float("inf")
    for epoch in range(args.epochs):
        losses = []
        model.train()
        for step, batch in enumerate(loader):
            if args.max_steps is not None and step >= args.max_steps:
                break
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            if not torch.isfinite(out.total_loss):
                raise RuntimeError(f"non-finite total loss at epoch={epoch} step={step}")
            out.total_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_value = float(out.total_loss.detach().cpu())
            losses.append(loss_value)
            total_steps = args.max_steps if args.max_steps is not None else len(loader)
            print(f"dynflow_swin_train epoch={epoch} step={step}/{total_steps} loss={loss_value:.4f}", flush=True)
        mean_loss = sum(losses) / max(len(losses), 1)
        metrics = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "steps": len(losses),
            "device": str(device),
            "synthetic": bool(args.synthetic),
            "note": "formal trainer uses the real BDD-X dataloader unless --synthetic is set",
        }
        (output_dir / f"metrics_epoch_{epoch:03d}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
        }
        epoch_ckpt = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(ckpt, epoch_ckpt)
        replace_link_or_copy(epoch_ckpt, output_dir / "checkpoint_latest.pth")
        if mean_loss < best_loss:
            best_loss = mean_loss
            replace_link_or_copy(epoch_ckpt, output_dir / "checkpoint_best_text.pth")
            replace_link_or_copy(epoch_ckpt, output_dir / "checkpoint_best_control.pth")
            replace_link_or_copy(epoch_ckpt, output_dir / "checkpoint_best_adapt_joint.pth")
            print(f"dynflow_swin_best epoch={epoch} train_loss={mean_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
