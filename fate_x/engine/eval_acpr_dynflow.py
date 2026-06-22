from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_x.engine.adapt_caption_eval_bridge import run_adapt_sep_caption_eval
from fate_x.acpr_dynflow.config import load_dynflow_config
from fate_x.acpr_dynflow.model import ACPRDynFlowModel
from fate_x.engine.acpr_dynflow_data import build_dynflow_dataloader


def _load_decode_tokenizer(cfg):
    bert_dir = cfg.raw.get("paths", {}).get("bert_dir", "models/captioning/bert-base-uncased")
    try:
        from src.layers.bert.tokenization_bert import BertTokenizer
        return BertTokenizer.from_pretrained(bert_dir, do_lower_case=True)
    except Exception:
        try:
            from transformers import BertTokenizer
            return BertTokenizer.from_pretrained(bert_dir, local_files_only=True)
        except Exception:
            return None


def _decode_ids(tokenizer, ids: list[int]) -> str:
    if tokenizer is None:
        return " ".join(str(int(x)) for x in ids if int(x) > 0)
    try:
        text = tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        tokens = [t for t in tokens if t not in {"[PAD]", "[CLS]", "[SEP]", "[MASK]"}]
        text = tokenizer.convert_tokens_to_string(tokens)
    return " ".join(str(text).replace("##", "").split())


def _pearson_or_none(x: torch.Tensor, y: torch.Tensor) -> tuple[float | None, str]:
    x = x.float().flatten()
    y = y.float().flatten()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 3:
        return None, "fewer_than_3_valid_points"
    x_std = x.std(unbiased=False)
    y_std = y.std(unbiased=False)
    if float(x_std.item()) <= 1e-8:
        return None, "x_near_constant"
    if float(y_std.item()) <= 1e-8:
        return None, "y_near_constant"
    corr = ((x - x.mean()) * (y - y.mean())).mean() / (x_std * y_std)
    return float(corr.clamp(-1, 1).item()), "ok"


def _std_or_none(x: torch.Tensor) -> float | None:
    x = x.float().flatten()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return None
    return float(x.std(unbiased=False).item())


def _collect_traffic_flow_audit(flow_speed_chunks, flow_course_chunks, pred_chunks, target_chunks) -> dict:
    audit = {
        "available": False,
        "sample_count": 0,
        "pred_speed_delta_corr": None,
        "pred_course_delta_corr": None,
        "true_speed_delta_corr": None,
        "true_course_delta_corr": None,
        "null_reasons": {},
    }
    if not flow_speed_chunks or not pred_chunks:
        audit["null_reasons"]["all"] = "no_flow_or_prediction_chunks"
        return audit
    flow_speed = torch.cat(flow_speed_chunks, dim=0)
    flow_course = torch.cat(flow_course_chunks, dim=0)
    pred_raw = torch.cat(pred_chunks, dim=0)
    audit["sample_count"] = int(pred_raw.shape[0])
    if pred_raw.ndim < 3 or pred_raw.shape[1] < 2 or pred_raw.shape[-1] < 2:
        audit["null_reasons"]["all"] = f"bad_prediction_shape_{tuple(pred_raw.shape)}"
        return audit
    pred_delta = pred_raw[:, 1:] - pred_raw[:, :-1]
    min_t = min(flow_speed.shape[1], pred_delta.shape[1])
    flow_speed = flow_speed[:, :min_t]
    flow_course = flow_course[:, :min_t]
    pred_delta = pred_delta[:, :min_t]
    audit["flow_speed_strength_std"] = _std_or_none(flow_speed)
    audit["flow_course_strength_std"] = _std_or_none(flow_course)
    audit["pred_speed_delta_std"] = _std_or_none(pred_delta[..., 1])
    audit["pred_course_delta_std"] = _std_or_none(pred_delta[..., 0])
    val, reason = _pearson_or_none(flow_speed, pred_delta[..., 1])
    audit["pred_speed_delta_corr"] = val
    if val is None:
        audit["null_reasons"]["pred_speed_delta_corr"] = reason
    val, reason = _pearson_or_none(flow_course, pred_delta[..., 0])
    audit["pred_course_delta_corr"] = val
    if val is None:
        audit["null_reasons"]["pred_course_delta_corr"] = reason
    if target_chunks:
        target_raw = torch.cat(target_chunks, dim=0)
        if target_raw.ndim >= 3 and target_raw.shape[1] >= 2 and target_raw.shape[-1] >= 2:
            target_delta = target_raw[:, 1:] - target_raw[:, :-1]
            min_t = min(min_t, target_delta.shape[1])
            valid_target = torch.isfinite(target_delta[:, :min_t]) & target_delta[:, :min_t].ne(-1.0)
            true_speed = torch.where(valid_target[..., 1], target_delta[:, :min_t, 1], torch.full_like(target_delta[:, :min_t, 1], float("nan")))
            true_course = torch.where(valid_target[..., 0], target_delta[:, :min_t, 0], torch.full_like(target_delta[:, :min_t, 0], float("nan")))
            audit["true_speed_delta_std"] = _std_or_none(true_speed)
            audit["true_course_delta_std"] = _std_or_none(true_course)
            val, reason = _pearson_or_none(flow_speed[:, :min_t], true_speed)
            audit["true_speed_delta_corr"] = val
            if val is None:
                audit["null_reasons"]["true_speed_delta_corr"] = reason
            val, reason = _pearson_or_none(flow_course[:, :min_t], true_course)
            audit["true_course_delta_corr"] = val
            if val is None:
                audit["null_reasons"]["true_course_delta_corr"] = reason
        else:
            audit["null_reasons"]["true_delta"] = f"bad_target_shape_{tuple(target_raw.shape)}"
    audit["available"] = True
    audit["has_null_prediction_corr"] = audit["pred_speed_delta_corr"] is None or audit["pred_course_delta_corr"] is None
    return audit


def _flow_strength_from_output(out) -> tuple[torch.Tensor, torch.Tensor]:
    contrib = out.ledger.factor_contributions_raw.detach().cpu()
    if contrib.ndim == 4 and contrib.shape[-1] >= 2:
        speed = contrib[:, 1:, :, 1].abs().sum(dim=-1)
        course = contrib[:, 1:, :, 0].abs().sum(dim=-1)
    else:
        probs = out.flow.factor_probs.detach().cpu()
        if probs.ndim == 2:
            probs = probs.unsqueeze(1)
        speed = probs[:, 1:].abs().mean(dim=-1) if probs.shape[1] > 1 else probs.abs().mean(dim=-1)
        course = speed.clone()
    lateral = out.flow.lateral_bias.detach().cpu().squeeze(-1)
    if lateral.ndim == 2 and lateral.shape[1] >= course.shape[1]:
        course = course + lateral[:, -course.shape[1]:].abs()
    return speed, course


def evaluate(config: str, checkpoint: str | None, output_dir: str, device: str = "cpu", max_samples: int = 8, synthetic: bool = False) -> dict:
    cfg = load_dynflow_config(config)
    loader = build_dynflow_dataloader(cfg.raw, "test", batch_size=1, max_samples=max_samples, synthetic=synthetic)
    model = ACPRDynFlowModel(cfg).to(device)
    if checkpoint and Path(checkpoint).exists():
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(data.get("model", data), strict=False)
    preds = []
    targets = []
    flow_speed_chunks = []
    flow_course_chunks = []
    prediction_rows = []
    tokenizer = _load_decode_tokenizer(cfg)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.frames = batch.frames.to(device)
            if batch.input_ids is not None:
                batch.input_ids = batch.input_ids.to(device)
            if batch.masked_ids is not None:
                batch.masked_ids = batch.masked_ids.to(device)
            if batch.control_target is not None:
                batch.control_target = batch.control_target.to(device)
            out = model(batch)
            pred_raw = out.ledger.final_prediction_raw.detach().cpu()
            preds.append(pred_raw)
            flow_speed, flow_course = _flow_strength_from_output(out)
            flow_speed_chunks.append(flow_speed)
            flow_course_chunks.append(flow_course)
            length = out.text.action_logits.shape[1]
            midpoint = max(1, length // 2)
            action_ids = out.text.action_logits[:, :midpoint].argmax(-1).detach().cpu().tolist()
            exp_ids = out.text.explanation_logits[:, midpoint:].argmax(-1).detach().cpu().tolist()
            for sid, des_ids, just_ids in zip(batch.sample_ids, action_ids, exp_ids):
                prediction_rows.append({"img_key": sid, "description": _decode_ids(tokenizer, des_ids), "explanation": _decode_ids(tokenizer, just_ids)})
            if batch.control_target is not None:
                targets.append(batch.control_target.detach().cpu())
    metrics = {"control_metrics_available": False}
    if preds and targets:
        metrics = model.codec.official_metrics(torch.cat(preds), torch.cat(targets))
        metrics["control_metrics_available"] = True
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if prediction_rows and not synthetic:
        text_metrics = run_adapt_sep_caption_eval(prediction_rows, loader, output_dir)
    else:
        text_metrics = {"text_metrics_available": False, "text_metrics_blocker": "synthetic eval or no generated prediction rows"}
    metrics.update(text_metrics)
    metrics["traffic_flow_audit"] = _collect_traffic_flow_audit(flow_speed_chunks, flow_course_chunks, preds, targets)
    metrics["description_empty_rate"] = sum(1 for r in prediction_rows if not r["description"]) / max(1, len(prediction_rows))
    metrics["explanation_empty_rate"] = sum(1 for r in prediction_rows if not r["explanation"]) / max(1, len(prediction_rows))
    (Path(output_dir) / "generated_caption_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prediction_rows), encoding="utf-8")
    (Path(output_dir) / "traffic_flow_audit.json").write_text(json.dumps(metrics["traffic_flow_audit"], indent=2), encoding="utf-8")
    (Path(output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_samples", type=int, default=8)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
