from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch

from fate_x.acpr_flow_v2.config import ACPRFlowCalV2Config, load_flowcal_v2_config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.acpr_flow_v2.types import FlowCalV2Batch
from fate_x.engine.acpr_flowcal_v2_data import build_v2_dataloader


def _synthetic_batch(batch_size: int, device: str, num_frames: int = 32, vocab: int = 101) -> FlowCalV2Batch:
    return FlowCalV2Batch(
        frames=torch.randn(batch_size, num_frames, 3, 224, 224, device=device),
        input_ids=torch.randint(0, vocab, (batch_size, 30), device=device),
        attention_mask=torch.ones(batch_size, 30, dtype=torch.long, device=device),
        token_type_ids=torch.zeros(batch_size, 30, dtype=torch.long, device=device),
        masked_pos=torch.tensor([[1, 2]] * batch_size, device=device),
        masked_ids=torch.randint(0, vocab, (batch_size, 2), device=device),
        car_info=torch.randn(batch_size, 2, num_frames, device=device),
        sample_ids=[f"memory_probe_synthetic_{idx}" for idx in range(batch_size)],
        raw_actions=["car slows down"] * batch_size,
        raw_justifications=["because traffic is ahead"] * batch_size,
    )


def _next_batch(loader_iter: Any, loader_factory, device: str) -> tuple[Any, FlowCalV2Batch]:
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader_factory())
        batch = next(loader_iter)
    for attr in ("frames", "input_ids", "attention_mask", "token_type_ids", "masked_pos", "masked_ids", "car_info"):
        value = getattr(batch, attr, None)
        if isinstance(value, torch.Tensor):
            setattr(batch, attr, value.to(device))
    return loader_iter, batch


def _measure_candidate(
    candidate: dict[str, int],
    device: str,
    synthetic: bool,
    config_path: str | None,
    synthetic_config: Optional[ACPRFlowCalV2Config],
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    batch_size = int(candidate["batch_size"])
    accum = int(candidate.get("gradient_accumulation_steps", 1))
    if synthetic:
        cfg = synthetic_config or ACPRFlowCalV2Config(
            hidden_dim=16,
            state_dim=16,
            text_hidden_dim=32,
            text_vocab_size=101,
            num_frames=4,
            use_real_video_swin=False,
        )
        model = ACPRFlowCalV2Model(cfg).to(device)
        loader_factory = None
        loader_iter = None
    else:
        cfg = load_flowcal_v2_config(config_path or "configs/acpr_flowcal_v2_bddx_32f_224.yaml")
        model = ACPRFlowCalV2Model(cfg).to(device)
        loader_factory = lambda: build_v2_dataloader("train", batch_size=batch_size, num_workers=0, config_path=config_path)
        loader_iter = iter(loader_factory())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    total_steps = warmup_steps + measured_steps
    finite = True
    blocker = None
    optimizer_steps = 0
    losses: list[float] = []
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(total_steps):
            if synthetic:
                batch = _synthetic_batch(batch_size, device, num_frames=cfg.num_frames, vocab=cfg.text_vocab_size)
            else:
                loader_iter, batch = _next_batch(loader_iter, loader_factory, device)
            out = model(batch, stage="M")
            loss = out.total_loss / max(1, accum)
            if not torch.isfinite(loss.detach()):
                finite = False
                blocker = "non-finite loss"
                break
            loss.backward()
            if (step + 1) % accum == 0 or step + 1 == total_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if step >= warmup_steps:
                losses.append(float(out.total_loss.detach().cpu()))
    except RuntimeError as exc:
        finite = False
        blocker = str(exc)
    peak_reserved = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_reserved = float(torch.cuda.max_memory_reserved() / (1024 ** 3))
    return {
        "candidate": candidate,
        "finite": bool(finite),
        "blocker": blocker,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "optimizer_steps": optimizer_steps,
        "loss_mean": sum(losses) / max(1, len(losses)),
        "peak_reserved_gib": peak_reserved,
        "device": device,
        "synthetic": synthetic,
    }


def run_probe(
    output_dir: str,
    candidates=None,
    device: str = "cuda",
    config_path: str | None = None,
    synthetic: bool = False,
    synthetic_config: Optional[ACPRFlowCalV2Config] = None,
    warmup_steps: int = 3,
    measured_steps: int = 30,
    hard_peak_reserved_limit_gib: float = 44.5,
) -> dict:
    candidates = candidates or [{"batch_size": 1, "gradient_accumulation_steps": 1}]
    records = []
    for candidate in candidates:
        record = _measure_candidate(candidate, device, synthetic, config_path, synthetic_config, warmup_steps, measured_steps)
        record["stable"] = bool(record["finite"] and record["optimizer_steps"] > 0 and record["peak_reserved_gib"] <= hard_peak_reserved_limit_gib)
        records.append(record)
    stable = [r for r in records if r["stable"]]
    selected = max(stable, key=lambda r: int(r["candidate"]["batch_size"]))["candidate"] if stable else None
    result = {
        "selected": selected,
        "candidates": records,
        "formal_cuda_gate": bool(device.startswith("cuda") and not synthetic),
        "review_pass_eligible": bool(selected is not None and device.startswith("cuda") and not synthetic),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("memory_probe.json", "memory_probe_selection.json"):
        (out / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_candidates(value: str | None) -> list[dict[str, int]] | None:
    if not value:
        return None
    out = []
    for part in value.split(","):
        batch, accum = part.split(":")
        out.append({"batch_size": int(batch), "gradient_accumulation_steps": int(accum)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--candidates", default=None, help="comma-separated batch:accum pairs, e.g. 4:8,3:11")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--measured_steps", type=int, default=30)
    args = p.parse_args()
    print(json.dumps(run_probe(
        args.output_dir,
        candidates=_parse_candidates(args.candidates),
        device=args.device,
        config_path=args.config,
        synthetic=args.synthetic,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
    ), indent=2))


if __name__ == "__main__":
    main()
