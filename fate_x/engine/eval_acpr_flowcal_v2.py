from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from fate_x.acpr_flow_v2.config import load_v2_config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.engine.adapt_caption_eval_bridge import run_adapt_sep_caption_eval
from fate_x.engine.acpr_flowcal_v2_data import adapt_batch_to_v2, build_v2_dataloader


CONTROL_THRESHOLDS = (0.1, 0.5, 1.0, 5.0, 10.0)


def shortest_circular_delta(pred: Tensor, target: Tensor, period: float = 360.0) -> Tensor:
    return torch.remainder(pred - target + period / 2.0, period) - period / 2.0


def compute_control_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    """Compute ADAPT-style continuous control metrics without discrete action proxies."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    course_err = shortest_circular_delta(prediction[..., 0], target[..., 0]).abs()
    speed_err = (prediction[..., 1] - target[..., 1]).abs()
    metrics: dict[str, float] = {
        "course_rmse": float(torch.sqrt((course_err ** 2).mean()).detach().cpu()),
        "speed_rmse": float(torch.sqrt((speed_err ** 2).mean()).detach().cpu()),
        "course_mae": float(course_err.mean().detach().cpu()),
        "speed_mae": float(speed_err.mean().detach().cpu()),
    }
    for threshold in CONTROL_THRESHOLDS:
        label = f"{threshold:g}"
        metrics[f"course_acc@{label}"] = float((course_err <= threshold).float().mean().detach().cpu())
        metrics[f"speed_acc@{label}"] = float((speed_err <= threshold).float().mean().detach().cpu())
    return metrics


def _pearson_or_none(x_values: list[float], y_values: list[float]) -> tuple[float | None, str | None]:
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return None, "insufficient_samples"
    x = torch.tensor(x_values, dtype=torch.float32)
    y = torch.tensor(y_values, dtype=torch.float32)
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2 or y.numel() < 2:
        return None, "insufficient_finite_samples"
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_energy = (x_centered ** 2).sum()
    y_energy = (y_centered ** 2).sum()
    if float(x_energy) <= 1e-12 and float(y_energy) <= 1e-12:
        return None, "both_zero_variance"
    if float(x_energy) <= 1e-12:
        return None, "traffic_factor_zero_variance"
    if float(y_energy) <= 1e-12:
        return None, "signal_zero_variance"
    denom = torch.sqrt(x_energy * y_energy)
    if float(denom) <= 1e-12:
        return None, "zero_variance"
    return float(((x_centered * y_centered).sum() / denom).detach().cpu()), None


def _std_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    x = torch.tensor(values, dtype=torch.float32)
    x = x[torch.isfinite(x)]
    if x.numel() < 2:
        return 0.0
    return float(x.std(unbiased=False).detach().cpu())


def _temporal_delta_signal(sequence: Tensor, axis: int) -> Tensor:
    if sequence.shape[1] < 2:
        return torch.zeros(sequence.shape[0], device=sequence.device, dtype=sequence.dtype)
    if axis == 0:
        delta = shortest_circular_delta(sequence[:, 1:, axis], sequence[:, :-1, axis]).abs()
    else:
        delta = (sequence[:, 1:, axis] - sequence[:, :-1, axis]).abs()
    return delta.mean(dim=1)


def _per_sample_diagnostic_signal(value: Any, batch_size: int, device: torch.device) -> Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=device, dtype=torch.float32)
    value = value.detach().to(device).float()
    if value.numel() == batch_size:
        return value.reshape(batch_size)
    if value.ndim >= 1 and value.shape[0] == batch_size:
        return value.reshape(batch_size, -1).mean(dim=1)
    return value.mean().expand(batch_size)


def _traffic_factor_candidates(diagnostics: dict[str, Any], batch_size: int, device: torch.device) -> dict[str, Tensor]:
    factors: dict[str, Tensor] = {}
    for key, value in diagnostics.items():
        if key.startswith("traffic_") and key.endswith("_per_sample"):
            name = key[: -len("_per_sample")]
            factors[name] = _per_sample_diagnostic_signal(value, batch_size, device)
    if not factors and "traffic_density" in diagnostics:
        factors["traffic_density"] = _per_sample_diagnostic_signal(diagnostics["traffic_density"], batch_size, device)
    return factors


def align_control_target(target_btc: Tensor, steps: int) -> Tensor:
    if target_btc.shape[1] == steps:
        return target_btc
    return torch.nn.functional.interpolate(target_btc.transpose(1, 2), size=steps, mode="linear", align_corners=False).transpose(1, 2)


def _move_batch_to_device(batch: Any, device: str):
    batch = adapt_batch_to_v2(batch)
    for attr in ("frames", "input_ids", "attention_mask", "token_type_ids", "masked_pos", "masked_ids", "car_info"):
        value = getattr(batch, attr, None)
        if isinstance(value, torch.Tensor):
            setattr(batch, attr, value.to(device))
    return batch


@torch.no_grad()
def evaluate(
    model,
    loader: Iterable,
    device: str = "cuda",
    max_batches: int = -1,
    stage: str = "S",
    output_dir: str | Path | None = None,
    tokenizer: Any = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    text_loss = 0.0
    count = 0
    control_predictions: list[Tensor] = []
    control_targets: list[Tensor] = []
    traffic_factor_series: dict[str, list[float]] = {}
    target_speed_delta_values: list[float] = []
    target_course_delta_values: list[float] = []
    pred_speed_delta_values: list[float] = []
    pred_course_delta_values: list[float] = []
    middle_rows: list[dict[str, Any]] = []
    caption_rows: list[dict[str, Any]] = []

    for batch_idx, raw_batch in enumerate(loader):
        if max_batches >= 0 and batch_idx >= max_batches:
            break
        batch = _move_batch_to_device(raw_batch, device)
        out = model(batch, stage=stage)
        total_loss += float(out.total_loss.detach().cpu())
        text_loss += float((out.action_text_loss + out.explanation_text_loss).detach().cpu())
        if batch.car_info is not None:
            target = batch.car_info.transpose(1, 2).to(out.control_final_prediction.device)
            target = align_control_target(target, out.control_final_prediction.shape[1])
            control_predictions.append(out.control_final_prediction.detach().cpu())
            control_targets.append(target.detach().cpu())
            traffic_factors = _traffic_factor_candidates(out.bundle.diagnostics, int(target.shape[0]), target.device)
            target_speed_delta = _temporal_delta_signal(target.detach(), axis=1)
            target_course_delta = _temporal_delta_signal(target.detach(), axis=0)
            pred_speed_delta = _temporal_delta_signal(out.control_final_prediction.detach(), axis=1)
            pred_course_delta = _temporal_delta_signal(out.control_final_prediction.detach(), axis=0)
            for factor_name, factor_values in traffic_factors.items():
                traffic_factor_series.setdefault(factor_name, []).extend([float(v.detach().cpu()) for v in factor_values])
            target_speed_delta_values.extend([float(v.detach().cpu()) for v in target_speed_delta])
            target_course_delta_values.extend([float(v.detach().cpu()) for v in target_course_delta])
            pred_speed_delta_values.extend([float(v.detach().cpu()) for v in pred_speed_delta])
            pred_course_delta_values.extend([float(v.detach().cpu()) for v in pred_course_delta])
        middle_rows.append(
            {
                "sample_ids": list(batch.sample_ids),
                "traffic_density": float(out.bundle.diagnostics.get("traffic_density", torch.tensor(0.0)).detach().cpu()),
                "transport_dustbin": float(out.bundle.diagnostics.get("transport_dustbin", torch.tensor(0.0)).detach().cpu()),
            }
        )
        if hasattr(model, "generate_adapt_caption_pairs"):
            try:
                caption_rows.extend(model.generate_adapt_caption_pairs(batch, tokenizer=tokenizer))
            except Exception as exc:
                caption_rows.append({"img_key": f"caption_generation_failed_{batch_idx}", "description": "", "explanation": str(exc)})
        count += 1

    count = max(1, count)
    metrics: dict[str, Any] = {
        "loss": total_loss / count,
        "text_loss": text_loss / count,
        "batches": count,
        "middle_output_preview": middle_rows[:8],
    }
    if output_dir is not None and caption_rows:
        metrics.update(run_adapt_sep_caption_eval(caption_rows, loader, output_dir))
    else:
        metrics.update(
            {
                "text_metrics_available": False,
                "text_metrics_blocker": "output_dir was not provided or no generated caption rows were produced; CIDEr is not inferred from loss.",
            }
        )
    if control_predictions:
        pred = torch.cat(control_predictions, dim=0)
        target = torch.cat(control_targets, dim=0)
        metrics.update(compute_control_metrics(pred, target))
        metrics["control_rmse"] = metrics["course_rmse"] + metrics["speed_rmse"]
        factor_audits: dict[str, dict[str, Any]] = {}
        for factor_name, factor_values in sorted(traffic_factor_series.items()):
            target_speed_corr, target_speed_reason = _pearson_or_none(factor_values, target_speed_delta_values)
            target_course_corr, target_course_reason = _pearson_or_none(factor_values, target_course_delta_values)
            pred_speed_corr, pred_speed_reason = _pearson_or_none(factor_values, pred_speed_delta_values)
            pred_course_corr, pred_course_reason = _pearson_or_none(factor_values, pred_course_delta_values)
            factor_audits[factor_name] = {
                "samples": len(factor_values),
                "std": _std_or_zero(factor_values),
                "target_speed_delta_corr": target_speed_corr,
                "target_speed_delta_corr_null_reason": target_speed_reason,
                "target_course_delta_corr": target_course_corr,
                "target_course_delta_corr_null_reason": target_course_reason,
                "pred_speed_delta_corr": pred_speed_corr,
                "pred_speed_delta_corr_null_reason": pred_speed_reason,
                "pred_course_delta_corr": pred_course_corr,
                "pred_course_delta_corr_null_reason": pred_course_reason,
            }
        if factor_audits:
            primary_factor = max(factor_audits, key=lambda name: factor_audits[name]["std"])
            primary_values = traffic_factor_series[primary_factor]
            primary_audit = factor_audits[primary_factor]
        else:
            primary_factor = "traffic_density"
            primary_values = []
            primary_audit = {
                "samples": 0,
                "std": 0.0,
                "target_speed_delta_corr": None,
                "target_speed_delta_corr_null_reason": "insufficient_samples",
                "target_course_delta_corr": None,
                "target_course_delta_corr_null_reason": "insufficient_samples",
                "pred_speed_delta_corr": None,
                "pred_speed_delta_corr_null_reason": "insufficient_samples",
                "pred_course_delta_corr": None,
                "pred_course_delta_corr_null_reason": "insufficient_samples",
            }
        metrics.update(
            {
                "target_speed_delta_corr": primary_audit["target_speed_delta_corr"],
                "target_course_delta_corr": primary_audit["target_course_delta_corr"],
                "pred_speed_delta_corr": primary_audit["pred_speed_delta_corr"],
                "pred_course_delta_corr": primary_audit["pred_course_delta_corr"],
                "traffic_flow_audit": {
                    "factor": primary_factor,
                    "primary_factor": primary_factor,
                    "samples": len(primary_values),
                    "traffic_factor_std": primary_audit["std"],
                    "target_speed_delta_std": _std_or_zero(target_speed_delta_values),
                    "target_course_delta_std": _std_or_zero(target_course_delta_values),
                    "pred_speed_delta_std": _std_or_zero(pred_speed_delta_values),
                    "pred_course_delta_std": _std_or_zero(pred_course_delta_values),
                    "target_speed_delta_corr": primary_audit["target_speed_delta_corr"],
                    "target_speed_delta_corr_null_reason": primary_audit["target_speed_delta_corr_null_reason"],
                    "target_course_delta_corr": primary_audit["target_course_delta_corr"],
                    "target_course_delta_corr_null_reason": primary_audit["target_course_delta_corr_null_reason"],
                    "pred_speed_delta_corr": primary_audit["pred_speed_delta_corr"],
                    "pred_speed_delta_corr_null_reason": primary_audit["pred_speed_delta_corr_null_reason"],
                    "pred_course_delta_corr": primary_audit["pred_course_delta_corr"],
                    "pred_course_delta_corr_null_reason": primary_audit["pred_course_delta_corr_null_reason"],
                    "factors": factor_audits,
                },
            }
        )
    else:
        metrics["control_metrics_available"] = False
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_v2_config(args.config)
    model = ACPRFlowCalV2Model(cfg).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    loader = build_v2_dataloader(args.split, batch_size=args.batch_size, config_path=args.config)
    metrics = evaluate(
        model,
        loader,
        args.device,
        args.max_batches,
        output_dir=Path(args.checkpoint).parent / "eval_cli",
        tokenizer=getattr(loader, "tokenizer", None),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
