from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from fate_x.engine.train_acpr_flowcal_pp import train_formal
from fate_x.utils.acpr_flow_artifacts import write_json
from fate_x.utils.acpr_flow_config import load_acpr_flow_config


def _cuda_memory() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated_gib": None, "reserved_gib": None}
    return {
        "allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def _default_candidates(cfg: dict[str, Any]) -> list[dict[str, int]]:
    return [
        {
            "batch_size": int(item["batch_size"]),
            "gradient_accumulation_steps": int(item["gradient_accumulation_steps"]),
        }
        for item in cfg.get("memory_probe", {}).get("candidates", [])
    ] or [
        {"batch_size": 5, "gradient_accumulation_steps": 13},
        {"batch_size": 4, "gradient_accumulation_steps": 16},
        {"batch_size": 3, "gradient_accumulation_steps": 22},
        {"batch_size": 2, "gradient_accumulation_steps": 32},
    ]


def run_memory_probe(
    config: str,
    output_dir: str | Path,
    device: str = "cuda",
    candidates: list[dict[str, int]] | None = None,
    warmup_steps: int | None = None,
    measured_steps: int | None = None,
) -> dict[str, Any]:
    cfg = load_acpr_flow_config(config) if Path(config).exists() else {"memory_probe": {}}
    probe_cfg = cfg.get("memory_probe", {})
    warmup_steps = int(warmup_steps if warmup_steps is not None else probe_cfg.get("warmup_steps", 3))
    measured_steps = int(measured_steps if measured_steps is not None else probe_cfg.get("measured_steps", 20))
    hard_limit = float(probe_cfg.get("hard_peak_reserved_limit_gib", 43))
    candidates = candidates or _default_candidates(cfg)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for item in candidates:
        batch_size = int(item["batch_size"])
        accum = int(item["gradient_accumulation_steps"])
        run_dir = out / f"memory_probe_b{batch_size}_a{accum}"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        record: dict[str, Any] = {
            "batch_size": batch_size,
            "gradient_accumulation_steps": accum,
            "effective_batch": batch_size * accum,
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "direct_image_training": True,
            "feature_cache_enabled": False,
            "returncode": 0,
        }
        try:
            train_formal(
                config,
                run_dir,
                device=device,
                max_steps=warmup_steps + measured_steps,
                batch_size=batch_size,
                epochs=1,
                load_pretrained_backbone=True,
            )
        except Exception as exc:  # pragma: no cover - exercised by remote/runtime gate.
            record["returncode"] = 1
            record["error"] = repr(exc)
        record["elapsed_seconds"] = time.time() - started
        record["memory"] = _cuda_memory()
        reserved = record["memory"]["reserved_gib"]
        record["stable"] = bool(record["returncode"] == 0 and (reserved is None or reserved <= hard_limit))
        results.append(record)
        if record["stable"] and selected is None:
            selected = record
            break

    report = {
        "direct_image_training": True,
        "feature_cache_enabled": False,
        "token_cache_enabled": False,
        "hard_peak_reserved_limit_gib": hard_limit,
        "candidates": results,
        "selected": selected,
    }
    write_json(out / "memory_probe_selection.json", report)
    write_json(out / "memory_probe.json", report)
    if selected is None:
        raise RuntimeError("No stable ACPR FlowCal++ memory-probe candidate completed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--measured_steps", type=int, default=None)
    args = parser.parse_args()
    report = run_memory_probe(
        config=args.config,
        output_dir=args.output_dir,
        device=args.device,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
    )
    print(report)


if __name__ == "__main__":
    main()
