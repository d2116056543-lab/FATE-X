from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fate_x.acpr_dynflow_swin.config import load_config
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.acpr_dynflow_swin.signal_codec import BDDXSignalCodec
from fate_x.engine.acpr_dynflow_swin_data import build_dynflow_swin_dataloader


def move_batch_to_device(batch: DynFlowSwinBatch, device: torch.device) -> DynFlowSwinBatch:
    batch.frames = batch.frames.to(device, non_blocking=True)
    batch.input_ids = batch.input_ids.to(device, non_blocking=True)
    batch.attention_mask = batch.attention_mask.to(device, non_blocking=True)
    batch.token_type_ids = batch.token_type_ids.to(device, non_blocking=True)
    batch.masked_pos = batch.masked_pos.to(device, non_blocking=True)
    batch.masked_ids = batch.masked_ids.to(device, non_blocking=True)
    if batch.control_target is not None:
        batch.control_target = batch.control_target.to(device, non_blocking=True)
    return batch


def _cider_sum(record: dict) -> float:
    return float(record.get("CIDEr_description", record.get("CIDEr_des", 0.0))) + float(
        record.get("CIDEr_explanation", record.get("CIDEr_exp", 0.0))
    )


def _control_score(record: dict, adapt_reference: dict) -> float:
    speed_ref = max(float(adapt_reference.get("speed_RMSE", 1.0)), 1e-6)
    course_ref = max(float(adapt_reference.get("course_RMSE", 1.0)), 1e-6)
    return 0.5 * float(record.get("speed_RMSE", 1e9)) / speed_ref + 0.5 * float(record.get("course_RMSE", 1e9)) / course_ref


def select_best_records(records: list[dict], adapt_reference: dict) -> dict[str, dict]:
    if not records:
        raise ValueError("records must not be empty")
    best_text = max(records, key=_cider_sum)
    best_control = min(records, key=lambda r: _control_score(r, adapt_reference))
    floor = 0.85 * float(adapt_reference.get("CIDEr_sum", 0.0))
    eligible = [record for record in records if _cider_sum(record) >= floor]
    pool = eligible or records
    best_test = min(
        pool,
        key=lambda r: (
            _control_score(r, adapt_reference),
            -float(r.get("CIDEr_explanation", r.get("CIDEr_exp", 0.0))),
            -float(r.get("CIDEr_description", r.get("CIDEr_des", 0.0))),
            float(r.get("speed_RMSE", 1e9)),
            float(r.get("course_RMSE", 1e9)),
        ),
    )
    return {
        "text": best_text,
        "control": best_control,
        "joint": best_test,
        "test": {**best_test, "text_floor_not_met": not bool(eligible)},
    }


def _load_decode_tokenizer(cfg: dict[str, Any]):
    bert_dir = cfg.get("paths", {}).get("bert_dir", "models/captioning/bert-base-uncased")
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
    ids = [int(token) for token in ids if int(token) > 0]
    if tokenizer is None:
        return " ".join(str(token) for token in ids)
    try:
        text = tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        tokens = [t for t in tokens if t not in {"[PAD]", "[CLS]", "[SEP]", "[MASK]"}]
        text = tokenizer.convert_tokens_to_string(tokens)
    return " ".join(str(text).replace("##", "").split())


def _flatten_control_metrics(control: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for signal_name, values in control.get("signals", {}).items():
        prefix = str(signal_name)
        for metric, value in values.items():
            key = f"{prefix}_{metric}".replace("@", "A")
            flat[key] = value
    return flat


def _maybe_text_metrics(prediction_rows: list[dict[str, str]], loader, output_dir: Path, synthetic: bool) -> dict[str, Any]:
    if synthetic:
        return {"text_metrics_available": False, "text_metrics_blocker": "synthetic evaluation"}
    if not prediction_rows:
        return {"text_metrics_available": False, "text_metrics_blocker": "no generated prediction rows"}
    try:
        from fate_x.engine.adapt_caption_eval_bridge import run_adapt_sep_caption_eval

        return run_adapt_sep_caption_eval(prediction_rows, loader, str(output_dir))
    except Exception as exc:
        return {
            "text_metrics_available": False,
            "text_metrics_blocker": f"ADAPT caption evaluation failed: {type(exc).__name__}: {exc}",
        }


def evaluate(
    config: str,
    checkpoint: str | None,
    output_dir: str,
    device: str = "cpu",
    max_samples: int = 8,
    synthetic: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    loader = build_dynflow_swin_dataloader(cfg, "test", batch_size=1, max_samples=max_samples, synthetic=synthetic)
    model = ACPRDynFlowSwinModel(cfg).to(device)
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)

    tokenizer = _load_decode_tokenizer(cfg)
    preds = []
    targets = []
    prediction_rows: list[dict[str, str]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, torch.device(device))
            out = model(batch, generate_text=True)
            preds.append(out.ledger.final_prediction_raw.detach().cpu())
            if batch.control_target is not None:
                targets.append(batch.control_target.detach().cpu())
            if out.text.generated_action is None or out.text.generated_explanation is None:
                raise RuntimeError("formal evaluation requires autoregressive generated text")
            for sample_id, description, explanation in zip(
                batch.sample_ids,
                out.text.generated_action,
                out.text.generated_explanation,
            ):
                prediction_rows.append(
                    {
                        "img_key": str(sample_id),
                        "description": description,
                        "explanation": explanation,
                    }
                )

    metrics: dict[str, Any] = {
        "sample_count": len(prediction_rows),
        "control_metrics_available": False,
        "text_metrics_available": False,
    }
    if preds and targets:
        control = BDDXSignalCodec().official_metrics(torch.cat(preds), torch.cat(targets))
        metrics["control_metrics"] = control
        metrics.update(_flatten_control_metrics(control))
        metrics["control_metrics_available"] = True
    metrics.update(_maybe_text_metrics(prediction_rows, loader, output_path, synthetic=synthetic))
    metrics["description_empty_rate"] = sum(1 for row in prediction_rows if not row["description"]) / max(1, len(prediction_rows))
    metrics["explanation_empty_rate"] = sum(1 for row in prediction_rows if not row["explanation"]) / max(1, len(prediction_rows))
    (output_path / "generated_caption_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows), encoding="utf-8"
    )
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
