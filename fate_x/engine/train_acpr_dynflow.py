from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_x.acpr_dynflow.config import load_dynflow_config
from fate_x.acpr_dynflow.model import ACPRDynFlowModel
from fate_x.engine.acpr_dynflow_data import build_dynflow_dataloader
from fate_x.engine.eval_acpr_dynflow import evaluate


def _move_batch(batch, device: str):
    for name in ("frames", "input_ids", "attention_mask", "token_type_ids", "masked_pos", "masked_ids", "control_target"):
        value = getattr(batch, name)
        if torch.is_tensor(value):
            setattr(batch, name, value.to(device))
    return batch


def train(config: str, output_dir: str, device: str = "cpu", batch_size: int | None = None, epochs: int | None = None, max_steps: int = -1, max_train_samples: int = -1, max_eval_samples: int = 8, synthetic: bool = False) -> None:
    cfg = load_dynflow_config(config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.json").write_text(json.dumps(cfg.raw, indent=2), encoding="utf-8")
    formal_epochs = int(cfg.get("optimization", "epochs", default=20)) if epochs is None else int(epochs)
    bs = int(batch_size or 1)
    loader = build_dynflow_dataloader(cfg.raw, "train", batch_size=bs, max_samples=max_train_samples, synthetic=synthetic)
    model = ACPRDynFlowModel(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, formal_epochs * max(1, len(loader))))
    best_text = -1.0
    best_control = float("inf")
    global_step = 0
    for epoch in range(formal_epochs):
        model.train()
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            out.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            global_step += 1
            record = {"event": "ACPR_DYNFLOW_BATCH", "epoch": epoch, "batch": batch_idx, "global_step": global_step, "loss": float(out.total_loss.detach().cpu()), "sample_ids": batch.sample_ids[:2], "frames_shape": list(batch.frames.shape)}
            print("ACPR_DYNFLOW_BATCH " + json.dumps(record, ensure_ascii=False), flush=True)
            with (out_dir / "loss_components.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**record, "loss_components": {k: float(v.detach().cpu()) for k, v in out.loss_components.items()}}, ensure_ascii=False) + "\n")
            if max_steps > 0 and global_step >= max_steps:
                break
        ckpt = {"model": model.state_dict(), "epoch": epoch, "global_step": global_step, "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict()}
        tmp = out_dir / "checkpoint_latest.pth.tmp"
        torch.save(ckpt, tmp)
        tmp.replace(out_dir / "checkpoint_latest.pth")
        eval_dir = out_dir / f"epoch_{epoch:03d}"
        metrics = evaluate(config=config, checkpoint=str(out_dir / "checkpoint_latest.pth"), output_dir=str(eval_dir), device=device, max_samples=max_eval_samples, synthetic=synthetic)
        speed_rmse = metrics.get("signals", {}).get("speed", {}).get("rmse")
        course_rmse = metrics.get("signals", {}).get("course", {}).get("rmse")
        control_score = float("inf") if speed_rmse is None or course_rmse is None else float(speed_rmse) + float(course_rmse)
        text_score = 0.0
        if text_score > best_text:
            best_text = text_score
            torch.save(ckpt, out_dir / "checkpoint_best_text.pth")
        if control_score < best_control:
            best_control = control_score
            torch.save(ckpt, out_dir / "checkpoint_best_control.pth")
            torch.save(ckpt, out_dir / "checkpoint_best_test.pth")
            torch.save(ckpt, out_dir / "checkpoint_best_joint.pth")
        with (out_dir / "metrics_summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "control_score": control_score, "metrics": metrics}, ensure_ascii=False) + "\n")
        if max_steps > 0 and global_step >= max_steps:
            break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=8)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()

