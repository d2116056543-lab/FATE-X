from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch

from fate_x.acpr_dynflow.config import load_dynflow_config
from fate_x.acpr_dynflow.model import ACPRDynFlowModel
from fate_x.engine.acpr_dynflow_data import build_dynflow_dataloader
from fate_x.engine.eval_acpr_dynflow import evaluate




def _current_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return ""


def _find_valid_review_pass(cfg) -> Path:
    head = _current_git_head()
    roots = []
    preflight_dir = cfg.raw.get("paths", {}).get("preflight_dir") or cfg.raw.get("preflight", {}).get("preflight_dir")
    if preflight_dir:
        roots.append(Path(preflight_dir))
    roots.extend(sorted(Path(".background_runs").glob("acpr_dynflow_v1_final_preflight*"), reverse=True))
    for root in roots:
        review = root / "review_report.json"
        pass_file = root / "REVIEW_PASS_ACPR_DYNFLOW_V1.txt"
        git_file = root / "git_provenance.json"
        if not (review.exists() and pass_file.exists() and git_file.exists()):
            continue
        try:
            report = json.loads(review.read_text(encoding="utf-8"))
            git = json.loads(git_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if report.get("passed") is True and not report.get("blockers") and git.get("head") == head and git.get("github_head") == head:
            return pass_file
    raise RuntimeError("ACPR-DynFlow formal training requires a clean REVIEW_PASS for the current GitHub-synced HEAD")

def _move_batch(batch, device: str):
    for name in ("frames", "input_ids", "attention_mask", "token_type_ids", "masked_pos", "masked_ids", "control_target"):
        value = getattr(batch, name)
        if torch.is_tensor(value):
            setattr(batch, name, value.to(device))
    return batch


def _resolve_gradient_accumulation(cfg, batch_size: int, explicit: int | None) -> int:
    if explicit is not None:
        if int(explicit) < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        return int(explicit)
    candidates = cfg.raw.get("memory_probe", {}).get("candidates", [])
    for item in candidates:
        if int(item.get("batch_size", -1)) == int(batch_size):
            return max(1, int(item.get("gradient_accumulation_steps", 1)))
    target = int(cfg.raw.get("memory_probe", {}).get("effective_batch_target", batch_size))
    return max(1, math.ceil(target / max(1, int(batch_size))))


def train(
    config: str,
    output_dir: str,
    device: str = "cpu",
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    epochs: int | None = None,
    max_steps: int = -1,
    max_train_samples: int = -1,
    max_eval_samples: int = -1,
    synthetic: bool = False,
) -> None:
    cfg = load_dynflow_config(config)
    review_pass = _find_valid_review_pass(cfg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review_pass_used.txt").write_text(str(review_pass), encoding="utf-8")
    (out_dir / "config_resolved.json").write_text(json.dumps(cfg.raw, indent=2), encoding="utf-8")
    if int(max_eval_samples) >= 0:
        eval_max_samples = int(max_eval_samples)
    else:
        eval_cfg = cfg.raw.get("evaluation", {})
        viz_cfg = cfg.raw.get("visualization", {})
        eval_max_samples = int(
            eval_cfg.get("best_checkpoint_cases", viz_cfg.get("best_checkpoint_cases", eval_cfg.get("lightweight_flow_audit_samples", -1)))
        )
    formal_epochs = int(cfg.get("optimization", "epochs", default=20)) if epochs is None else int(epochs)
    first_candidate = cfg.raw.get("memory_probe", {}).get("candidates", [{}])[0]
    bs = int(batch_size or first_candidate.get("batch_size") or 1)
    accum_steps = _resolve_gradient_accumulation(cfg, bs, gradient_accumulation_steps)
    loader = build_dynflow_dataloader(cfg.raw, "train", batch_size=bs, max_samples=max_train_samples, synthetic=synthetic)
    model = ACPRDynFlowModel(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    optimizer_steps_per_epoch = max(1, math.ceil(max(1, len(loader)) / accum_steps))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, formal_epochs * optimizer_steps_per_epoch))
    best_text = -1.0
    best_control = float("inf")
    global_step = 0
    optimizer_step = 0
    (out_dir / "training_effective_batch.json").write_text(
        json.dumps(
            {
                "batch_size": bs,
                "gradient_accumulation_steps": accum_steps,
                "effective_batch_size": bs * accum_steps,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "source": "cli_or_memory_probe_config",
                "eval_max_samples": eval_max_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for epoch in range(formal_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            out = model(batch)
            scaled_loss = out.total_loss / accum_steps
            scaled_loss.backward()
            global_step += 1
            is_last_batch = batch_idx == len(loader) - 1
            reached_max_steps = max_steps > 0 and global_step >= max_steps
            should_step = ((batch_idx + 1) % accum_steps == 0) or is_last_batch or reached_max_steps
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                optimizer_step += 1
            record = {
                "event": "ACPR_DYNFLOW_BATCH",
                "epoch": epoch,
                "batch": batch_idx,
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "gradient_accumulation_steps": accum_steps,
                "optimizer_stepped": should_step,
                "loss": float(out.total_loss.detach().cpu()),
                "loss_scaled_for_backward": float(scaled_loss.detach().cpu()),
                "sample_ids": batch.sample_ids[:2],
                "frames_shape": list(batch.frames.shape),
            }
            print("ACPR_DYNFLOW_BATCH " + json.dumps(record, ensure_ascii=False), flush=True)
            with (out_dir / "loss_components.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({**record, "loss_components": {k: float(v.detach().cpu()) for k, v in out.loss_components.items()}}, ensure_ascii=False) + "\n")
            if reached_max_steps:
                break
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "gradient_accumulation_steps": accum_steps,
            "effective_batch_size": bs * accum_steps,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        tmp = out_dir / "checkpoint_latest.pth.tmp"
        torch.save(ckpt, tmp)
        tmp.replace(out_dir / "checkpoint_latest.pth")
        eval_dir = out_dir / f"epoch_{epoch:03d}"
        metrics = evaluate(config=config, checkpoint=str(out_dir / "checkpoint_latest.pth"), output_dir=str(eval_dir), device=device, max_samples=eval_max_samples, synthetic=synthetic)
        speed_rmse = metrics.get("signals", {}).get("speed", {}).get("rmse")
        course_rmse = metrics.get("signals", {}).get("course", {}).get("rmse")
        control_score = float("inf") if speed_rmse is None or course_rmse is None else float(speed_rmse) + float(course_rmse)
        text_score = metrics.get("CIDEr_des+exp") if metrics.get("text_metrics_available") else None
        if text_score is not None and float(text_score) > best_text:
            best_text = float(text_score)
            torch.save(ckpt, out_dir / "checkpoint_best_text.pth")
        if control_score < best_control:
            best_control = control_score
            torch.save(ckpt, out_dir / "checkpoint_best_control.pth")
            torch.save(ckpt, out_dir / "checkpoint_best_test.pth")
            if text_score is not None:
                torch.save(ckpt, out_dir / "checkpoint_best_joint.pth")
        with (out_dir / "metrics_summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "control_score": control_score, "text_score": text_score, "metrics": metrics}, ensure_ascii=False) + "\n")
        traffic_audit = metrics.get("traffic_flow_audit", {})
        print("ACPR_DYNFLOW_EVAL " + json.dumps({"epoch": epoch, "control_score": control_score, "text_score": text_score, "text_metrics_available": metrics.get("text_metrics_available"), "text_metrics_blocker": metrics.get("text_metrics_blocker"), "traffic_flow_audit": traffic_audit}, ensure_ascii=False), flush=True)
        if max_steps > 0 and global_step >= max_steps:
            break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
