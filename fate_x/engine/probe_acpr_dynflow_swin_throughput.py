from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from fate_x.acpr_dynflow_swin.config import load_config
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.engine.acpr_dynflow_swin_data import build_dynflow_swin_dataloader
from fate_x.engine.train_acpr_dynflow_swin import build_optimizer_groups, move_batch_to_device


def select_best_candidate(
    candidates: list[dict[str, Any]],
    max_epoch_hours: float,
    max_peak_reserved_gib: float,
    max_data_time_fraction: float,
) -> dict[str, Any] | None:
    viable: list[dict[str, Any]] = []
    for candidate in candidates:
        finite = candidate.get("finite") is True
        memory_ok = float(candidate.get("peak_reserved_gib", float("inf"))) <= max_peak_reserved_gib
        epoch_ok = float(candidate.get("projected_train_epoch_hours", float("inf"))) <= max_epoch_hours
        data_ok = float(candidate.get("data_time_fraction", 1.0)) <= max_data_time_fraction
        optimizer_ok = int(candidate.get("optimizer_steps", 0)) > 0
        candidate["formal_gate_passed"] = finite and memory_ok and epoch_ok and data_ok and optimizer_ok
        candidate["gate_checks"] = {
            "finite": finite,
            "memory_ok": memory_ok,
            "epoch_time_ok": epoch_ok,
            "data_time_ok": data_ok,
            "optimizer_steps_ok": optimizer_ok,
        }
        if candidate["formal_gate_passed"]:
            viable.append(candidate)
    if not viable:
        return None
    selected = max(viable, key=lambda item: float(item.get("samples_per_second", 0.0)))
    selected = dict(selected)
    selected["selection_reason"] = "highest_samples_per_second_under_all_gates"
    return selected


def filter_candidate_configs(candidates: list[dict[str, Any]], batch_sizes: str | None) -> list[dict[str, Any]]:
    if not batch_sizes:
        return list(candidates)
    requested = {int(part.strip()) for part in batch_sizes.split(",") if part.strip()}
    return [candidate for candidate in candidates if int(candidate["batch_size"]) in requested]


def _memory_gib(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0}
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
    }


def should_abort_after_warmup_memory(memory: dict[str, float], max_peak_reserved_gib: float) -> bool:
    return float(memory.get("peak_reserved_gib", 0.0)) > float(max_peak_reserved_gib)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_candidate(
    cfg: dict[str, Any],
    device: torch.device,
    batch_size: int,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    measured_steps: int,
    max_samples: int,
    synthetic: bool,
    max_peak_reserved_gib: float | None = None,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = ACPRDynFlowSwinModel(cfg).to(device)
    optimizer = torch.optim.AdamW(build_optimizer_groups(model, cfg))
    loader = build_dynflow_swin_dataloader(
        cfg,
        split="train",
        batch_size=batch_size,
        max_samples=max(max_samples, batch_size * (warmup_steps + measured_steps + 2)),
        synthetic=synthetic,
    )
    iterator = iter(loader)
    use_bf16 = device.type == "cuda" and cfg.get("optimization", {}).get("precision") == "bf16"
    finite = True
    skipped_optimizer_step = False
    data_time = forward_time = backward_time = optimizer_time = 0.0
    measured_batches = 0
    optimizer_steps = 0

    def one_step(is_measured: bool, index: int) -> None:
        nonlocal data_time, forward_time, backward_time, optimizer_time
        nonlocal finite, skipped_optimizer_step, measured_batches, optimizer_steps
        t0 = time.perf_counter()
        batch = move_batch_to_device(next(iterator), device)
        _sync(device)
        t1 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            output = model(batch)
            loss = output.total_loss / max(gradient_accumulation_steps, 1)
        _sync(device)
        t2 = time.perf_counter()
        if not torch.isfinite(loss):
            finite = False
        loss.backward()
        _sync(device)
        t3 = time.perf_counter()
        if (index + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(cfg.get("optimization", {}).get("gradient_clip_norm", 1.0)),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        _sync(device)
        t4 = time.perf_counter()
        if is_measured:
            data_time += t1 - t0
            forward_time += t2 - t1
            backward_time += t3 - t2
            optimizer_time += t4 - t3
            measured_batches += 1
            if (index + 1) % gradient_accumulation_steps == 0 and optimizer_steps == 0:
                skipped_optimizer_step = True

    model.train()
    try:
        for i in range(warmup_steps):
            one_step(False, i)
        warmup_memory = _memory_gib(device)
        if max_peak_reserved_gib is not None and should_abort_after_warmup_memory(warmup_memory, max_peak_reserved_gib):
            return {
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "warmup_steps": warmup_steps,
                "measured_steps": 0,
                "optimizer_steps": optimizer_steps,
                "passed": False,
                "finite": True,
                "formal_gate_passed": False,
                "abort_reason": "warmup_peak_reserved_exceeds_hard_cap",
                "synthetic": bool(synthetic),
                "device": str(device),
                **warmup_memory,
            }
        _sync(device)
        start = time.perf_counter()
        for i in range(measured_steps):
            one_step(True, i)
        _sync(device)
        elapsed = time.perf_counter() - start
    except StopIteration:
        finite = False
        elapsed = max(data_time + forward_time + backward_time + optimizer_time, 1e-6)
    except RuntimeError as exc:
        return {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "passed": False,
            "finite": False,
            "error": str(exc),
            **_memory_gib(device),
        }
    train_count = len(getattr(loader, "dataset", [])) if getattr(loader, "dataset", None) is not None else 0
    samples_per_second = measured_batches * batch_size / max(elapsed, 1e-6)
    projected_hours = train_count / max(samples_per_second, 1e-6) / 3600.0 if train_count else float("inf")
    total_measured_time = max(data_time + forward_time + backward_time + optimizer_time, 1e-6)
    component_times = {
        "data_time": data_time,
        "forward_time": forward_time,
        "predicate_traffic_consolidation_motion_text_time": forward_time,
        "backward_time": backward_time,
        "optimizer_time": optimizer_time,
    }
    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_batches,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples_per_second,
        "projected_train_epoch_hours": projected_hours,
        "projected_test_eval_epoch_hours": None,
        "data_time_fraction": data_time / total_measured_time,
        "finite": bool(finite),
        "skipped_optimizer_step": bool(skipped_optimizer_step),
        "bf16_autocast_enabled": bool(use_bf16),
        "component_times": component_times,
        "synthetic": bool(synthetic),
        "device": str(device),
        **_memory_gib(device),
    }


def run_probe(
    config: str,
    output: str | None,
    device_name: str | None,
    synthetic: bool,
    quick_steps: int | None,
    candidate_batch_sizes: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    probe_cfg = cfg.get("memory_throughput_probe", {})
    warmup_steps = int(probe_cfg.get("warmup_steps", 10))
    measured_steps = int(probe_cfg.get("measured_steps", 100))
    if quick_steps is not None:
        warmup_steps = min(warmup_steps, 1)
        measured_steps = int(quick_steps)
    candidates_cfg = filter_candidate_configs(
        probe_cfg.get("candidates", [{"batch_size": 1, "gradient_accumulation_steps": 1}]),
        candidate_batch_sizes,
    )
    candidates: list[dict[str, Any]] = []
    max_samples = max(256, max(int(c["batch_size"]) for c in candidates_cfg) * (warmup_steps + measured_steps + 2))
    output_path = Path(output) if output else None
    for candidate_cfg in candidates_cfg:
        candidate = measure_candidate(
            cfg,
            device,
            batch_size=int(candidate_cfg["batch_size"]),
            gradient_accumulation_steps=int(candidate_cfg["gradient_accumulation_steps"]),
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            max_samples=max_samples,
            synthetic=synthetic,
            max_peak_reserved_gib=float(probe_cfg.get("hard_peak_reserved_limit_gib", 44.0)),
        )
        candidates.append(candidate)
        if output_path:
            partial_selected = select_best_candidate(
                candidates,
                float(probe_cfg.get("projected_train_epoch_limit_hours", 4.0)),
                float(probe_cfg.get("hard_peak_reserved_limit_gib", 44.0)),
                float(probe_cfg.get("maximum_data_time_fraction", 0.20)),
            )
            partial_report = {
                "status": "partial_pass" if partial_selected else "partial_blocked",
                "passed": False,
                "selected_candidate": partial_selected,
                "candidate_reports": candidates,
                "formal_probe": quick_steps is None,
                "partial": True,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(partial_report, indent=2), encoding="utf-8")
    max_epoch_hours = float(probe_cfg.get("projected_train_epoch_limit_hours", 4.0))
    max_peak = float(probe_cfg.get("hard_peak_reserved_limit_gib", 44.0))
    max_data = float(probe_cfg.get("maximum_data_time_fraction", 0.20))
    selected = select_best_candidate(candidates, max_epoch_hours, max_peak, max_data)
    report = {
        "status": "pass" if selected else "blocked",
        "passed": selected is not None,
        "selection_metric": probe_cfg.get("selection_metric", "highest_samples_per_second"),
        "selected_candidate": selected,
        "candidate_reports": candidates,
        "limits": {
            "projected_train_epoch_limit_hours": max_epoch_hours,
            "hard_peak_reserved_limit_gib": max_peak,
            "maximum_data_time_fraction": max_data,
        },
        "formal_probe": quick_steps is None,
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--quick_steps", type=int, default=None, help="development smoke only; formal probe omits this")
    parser.add_argument("--candidate_batch_sizes", default=None)
    args = parser.parse_args()
    print(json.dumps(run_probe(
        args.config,
        args.output,
        args.device,
        args.synthetic,
        args.quick_steps,
        candidate_batch_sizes=args.candidate_batch_sizes,
    ), indent=2))


if __name__ == "__main__":
    main()
