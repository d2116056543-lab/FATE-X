from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch

from fate_x.acpr_dynflow_swin.config import load_config
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.acpr_dynflow_swin.signal_codec import BDDXSignalCodec
from fate_x.acpr_dynflow_swin.types import DynFlowSwinBatch
from fate_x.engine.acpr_dynflow_swin_data import build_dynflow_swin_dataloader
from fate_x.engine.eval_acpr_dynflow_swin import evaluate, select_best_records


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


def build_linear_warmup_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    total_optimizer_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total = max(int(total_optimizer_steps), 1)
    warmup = max(int(round(total * float(warmup_ratio))), 1)
    floor = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        step = max(int(step), 0)
        if step <= warmup:
            return max(floor, step / float(warmup))
        progress = min(1.0, (step - warmup) / float(max(total - warmup, 1)))
        return max(floor, 1.0 - (1.0 - floor) * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _torch_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python_random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def build_training_state(
    epoch: int,
    global_step: int,
    optimizer_step: int,
    gradient_accumulation_steps: int,
    best_records: dict[str, Any],
    signal_codec: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer_step": int(optimizer_step),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "rng_state": _torch_rng_state(),
        "best_records": best_records,
        "signal_codec": signal_codec,
        "config": config,
    }


def atomic_save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    if target.name.endswith(".tmp"):
        raise ValueError(f"refusing to write checkpoint directly to .tmp path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_adapt_reference_metrics(cfg: dict[str, Any]) -> dict[str, float]:
    paths = cfg.get("paths", {})
    eval_cfg = cfg.get("evaluation", {})
    control_ref = eval_cfg.get("adapt_reference_control", {})
    def _read_cider(key: str) -> float:
        path = Path(paths.get(key, ""))
        if not path.exists():
            return 0.0
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload.get("CIDEr", 0.0))

    cider_des = _read_cider("adapt_reference_description_metrics")
    cider_exp = _read_cider("adapt_reference_explanation_metrics")
    return {
        "CIDEr_description": cider_des,
        "CIDEr_explanation": cider_exp,
        "CIDEr_sum": cider_des + cider_exp,
        "speed_RMSE": float(control_ref.get("speed_RMSE", 2.68)),
        "course_RMSE": float(control_ref.get("course_RMSE", 5.87)),
    }


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


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if state.get("python_random") is not None:
        random.setstate(state["python_random"])
    if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


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
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("optimization", {}).get("tf32", True))
        torch.backends.cudnn.benchmark = bool(cfg.get("optimization", {}).get("cudnn_benchmark", True))
    model = ACPRDynFlowSwinModel(cfg).to(device)
    optimizer = torch.optim.AdamW(build_optimizer_groups(model, cfg))
    batch_size = args.batch_size or int(cfg.get("optimization", {}).get("batch_size", 1))
    grad_accum = int(args.gradient_accumulation_steps or cfg.get("optimization", {}).get("gradient_accumulation_steps", 1))
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
    steps_per_epoch = args.max_steps if args.max_steps is not None else len(loader)
    total_optimizer_steps = max(1, ((max(steps_per_epoch, 1) + grad_accum - 1) // grad_accum) * args.epochs)
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer,
        total_optimizer_steps=total_optimizer_steps,
        warmup_ratio=float(cfg.get("optimization", {}).get("warmup_ratio", 0.1)),
        min_lr_ratio=float(cfg.get("optimization", {}).get("min_lr_ratio", 0.0)),
    )
    eval_records: list[dict] = []
    best_records: dict[str, Any] = {}
    global_step = 0
    optimizer_step = 0
    start_epoch = 0
    adapt_reference = load_adapt_reference_metrics(cfg)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        _restore_rng_state(state.get("rng_state"))
        start_epoch = int(state.get("epoch", -1)) + 1
        global_step = int(state.get("global_step", 0))
        optimizer_step = int(state.get("optimizer_step", 0))
        best_records = dict(state.get("best_records", {}))
    for epoch in range(start_epoch, args.epochs):
        losses = []
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            if args.max_steps is not None and step >= args.max_steps:
                break
            batch = move_batch_to_device(batch, device)
            use_bf16 = device.type == "cuda" and cfg.get("optimization", {}).get("precision") == "bf16"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(batch)
                scaled_loss = out.total_loss / max(grad_accum, 1)
            if not torch.isfinite(scaled_loss):
                raise RuntimeError(f"non-finite total loss at epoch={epoch} step={step}")
            scaled_loss.backward()
            do_step = ((step + 1) % grad_accum == 0) or (step + 1 == steps_per_epoch)
            if do_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(cfg.get("optimization", {}).get("gradient_clip_norm", 1.0)),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            global_step += 1
            loss_value = float(out.total_loss.detach().cpu())
            losses.append(loss_value)
            total_steps = steps_per_epoch
            print(f"dynflow_swin_train epoch={epoch} step={step}/{total_steps} loss={loss_value:.4f}", flush=True)
        mean_loss = sum(losses) / max(len(losses), 1)
        metrics = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "steps": len(losses),
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "gradient_accumulation_steps": grad_accum,
            "lr": [group["lr"] for group in optimizer.param_groups],
            "device": str(device),
            "synthetic": bool(args.synthetic),
            "note": "formal trainer uses the real BDD-X dataloader unless --synthetic is set",
        }
        (output_dir / f"metrics_epoch_{epoch:03d}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            **build_training_state(
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                gradient_accumulation_steps=grad_accum,
                best_records=best_records,
                signal_codec=BDDXSignalCodec().state_dict(),
                config=cfg,
            ),
        }
        epoch_ckpt = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        atomic_save_checkpoint(epoch_ckpt, ckpt)
        replace_link_or_copy(epoch_ckpt, output_dir / "checkpoint_latest.pth")
        eval_dir = output_dir / f"epoch_{epoch:03d}" / "full_test"
        eval_metrics = evaluate(
            config=args.config,
            checkpoint=str(epoch_ckpt),
            output_dir=str(eval_dir),
            device=str(device),
            max_samples=args.eval_max_samples,
            synthetic=args.synthetic,
        )
        record = {"epoch": epoch, **eval_metrics}
        eval_records.append(record)
        (output_dir / "eval_records.jsonl").open("a", encoding="utf-8").write(json.dumps(record, ensure_ascii=False) + "\n")
        best = select_best_records(eval_records, adapt_reference=adapt_reference)
        best_records = best
        if best["text"]["epoch"] == epoch:
            replace_link_or_copy(epoch_ckpt, output_dir / "checkpoint_best_text.pth")
        if best["control"]["epoch"] == epoch:
            replace_link_or_copy(epoch_ckpt, output_dir / "checkpoint_best_control.pth")
        if best["joint"]["epoch"] == epoch:
            replace_link_or_copy(epoch_ckpt, output_dir / "checkpoint_best_joint.pth")
        if best["test"]["epoch"] == epoch:
            replace_link_or_copy(epoch_ckpt, output_dir / "checkpoint_best_test.pth")
        (output_dir / "best_records.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(
            "dynflow_swin_full test "
            + json.dumps(
                {
                    "epoch": epoch,
                    "CIDEr_description": eval_metrics.get("CIDEr_description"),
                    "CIDEr_explanation": eval_metrics.get("CIDEr_explanation"),
                    "speed_RMSE": eval_metrics.get("speed_RMSE"),
                    "course_RMSE": eval_metrics.get("course_RMSE"),
                    "best": {k: v.get("epoch") for k, v in best.items()},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
