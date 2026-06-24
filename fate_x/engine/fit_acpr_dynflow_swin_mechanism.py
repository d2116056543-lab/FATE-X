from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fate_x.acpr_dynflow_swin.config import load_config
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.engine.acpr_dynflow_swin_data import build_dynflow_swin_dataloader
from fate_x.engine.eval_acpr_dynflow_swin import move_batch_to_device
from fate_x.engine.train_acpr_dynflow_swin import build_optimizer_groups


REQUIRED_IMPROVEMENTS = (
    "total_loss",
    "final_speed_normalized",
    "final_course_normalized",
    "action_text",
    "explanation_text",
    "predicate_nnpu",
    "pattern_semantic",
    "traffic_state_semantic",
    "contribution_alignment",
)


def summarize_mechanism_fit(
    initial: dict[str, float],
    final: dict[str, float],
    collapse_stats: dict[str, float],
    sample_count: int,
    optimizer_steps: int,
    min_improvement: float = 0.0,
) -> dict[str, Any]:
    improvements = {
        name: float(initial.get(name, 0.0)) - float(final.get(name, 0.0))
        for name in REQUIRED_IMPROVEMENTS
    }
    failed_improvements = [
        name for name, value in improvements.items()
        if value <= float(min_improvement)
    ]
    collapse_checks = {
        "predicates_not_identical": float(collapse_stats.get("predicate_std", 0.0)) > 1e-4,
        "pattern_not_constant": float(collapse_stats.get("pattern_std", 0.0)) > 1e-4,
        "factors_not_constant": float(collapse_stats.get("factor_std", 0.0)) > 1e-4,
        "flow_contributions_nonzero": float(collapse_stats.get("flow_contribution_mean_abs", 0.0)) > 1e-6,
        "benefit_gates_nonzero": float(collapse_stats.get("benefit_gate_mean_abs", 0.0)) > 1e-6,
        "course_uses_lateral_factors": float(collapse_stats.get("course_lateral_effect_mean_abs", 0.0)) > 1e-6,
    }
    failed_collapse_checks = [name for name, ok in collapse_checks.items() if not ok]
    passed = (
        int(sample_count) >= 128
        and int(optimizer_steps) > 0
        and not failed_improvements
        and not failed_collapse_checks
    )
    return {
        "status": "pass" if passed else "blocked",
        "passed": passed,
        "sample_count": int(sample_count),
        "optimizer_steps": int(optimizer_steps),
        "initial": initial,
        "final": final,
        "improvements": improvements,
        "failed_improvements": failed_improvements,
        "collapse_stats": collapse_stats,
        "collapse_checks": collapse_checks,
        "failed_collapse_checks": failed_collapse_checks,
        "required_improvements": list(REQUIRED_IMPROVEMENTS),
    }


def _component_snapshot(output) -> dict[str, float]:
    values = {
        name: float(value.detach().float().cpu())
        for name, value in output.loss_components.items()
        if not name.startswith("raw/")
    }
    values["total_loss"] = float(output.total_loss.detach().float().cpu())
    return values


def _merge_average(items: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {
        key: sum(float(item.get(key, 0.0)) for item in items) / max(len(items), 1)
        for key in keys
    }


def _collapse_stats(output) -> dict[str, float]:
    gated = output.ledger.gated_factor_contributions_normalized.detach().float()
    lateral = gated[:, :, [2, 3], 0] if gated.shape[2] > 3 else gated[:, :, :, 0]
    return {
        "predicate_std": float(output.predicates.probabilities.detach().float().std().cpu()),
        "pattern_std": float(output.traffic.pattern_probs.detach().float().std().cpu()),
        "factor_std": float(output.traffic.factor_probs.detach().float().std().cpu()),
        "flow_contribution_mean_abs": float(gated.abs().mean().cpu()),
        "benefit_gate_mean_abs": float(output.ledger.benefit_gate.detach().float().abs().mean().cpu()),
        "course_lateral_effect_mean_abs": float(lateral.abs().mean().cpu()),
    }


def run_mechanism_fit(
    config: str,
    output: str,
    device_name: str | None = None,
    sample_count: int = 128,
    batch_size: int = 1,
    max_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    cfg = load_config(config)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(20260625)
    model = ACPRDynFlowSwinModel(cfg).to(device)
    optimizer = torch.optim.AdamW(build_optimizer_groups(model, cfg))
    loader = build_dynflow_swin_dataloader(
        cfg,
        split="train",
        batch_size=batch_size,
        max_samples=sample_count,
        synthetic=False,
    )
    model.train()
    use_bf16 = device.type == "cuda" and cfg.get("optimization", {}).get("precision") == "bf16"
    optimizer_steps = 0
    seen = 0
    snapshots: list[dict[str, float]] = []
    collapse: dict[str, float] = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            model_output = model(batch)
        if not torch.isfinite(model_output.total_loss):
            raise RuntimeError(f"non-finite mechanism-fit loss at optimizer_step={optimizer_steps}")
        snapshots.append(_component_snapshot(model_output))
        collapse = _collapse_stats(model_output)
        model_output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(cfg.get("optimization", {}).get("gradient_clip_norm", 1.0)),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        seen += len(batch.sample_ids)
        if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
            break
        if seen >= sample_count:
            break
    window = max(1, min(4, len(snapshots) // 4))
    initial = _merge_average(snapshots[:window])
    final = _merge_average(snapshots[-window:])
    report = summarize_mechanism_fit(
        initial,
        final,
        collapse,
        sample_count=seen,
        optimizer_steps=optimizer_steps,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample_count", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_optimizer_steps", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_mechanism_fit(
        config=args.config,
        output=args.output,
        device_name=args.device,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        max_optimizer_steps=args.max_optimizer_steps,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
