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


def evaluate(config: str, checkpoint: str | None, output_dir: str, device: str = "cpu", max_samples: int = 8, synthetic: bool = False) -> dict:
    cfg = load_dynflow_config(config)
    loader = build_dynflow_dataloader(cfg.raw, "test", batch_size=1, max_samples=max_samples, synthetic=synthetic)
    model = ACPRDynFlowModel(cfg).to(device)
    if checkpoint and Path(checkpoint).exists():
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(data.get("model", data), strict=False)
    preds = []
    targets = []
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
            preds.append(out.ledger.final_prediction_raw.cpu())
            length = out.text.action_logits.shape[1]
            midpoint = max(1, length // 2)
            action_ids = out.text.action_logits[:, :midpoint].argmax(-1).detach().cpu().tolist()
            exp_ids = out.text.explanation_logits[:, midpoint:].argmax(-1).detach().cpu().tolist()
            for sid, des_ids, just_ids in zip(batch.sample_ids, action_ids, exp_ids):
                prediction_rows.append({"img_key": sid, "description": _decode_ids(tokenizer, des_ids), "explanation": _decode_ids(tokenizer, just_ids)})
            if batch.control_target is not None:
                targets.append(batch.control_target.cpu())
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
    metrics["description_empty_rate"] = sum(1 for r in prediction_rows if not r["description"]) / max(1, len(prediction_rows))
    metrics["explanation_empty_rate"] = sum(1 for r in prediction_rows if not r["explanation"]) / max(1, len(prediction_rows))
    (Path(output_dir) / "generated_caption_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prediction_rows), encoding="utf-8")
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

