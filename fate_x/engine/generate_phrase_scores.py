from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fate_x.explain.phrase_attribution import find_phrase_hits
from fate_x.explain.phrase_counterfactual import score_drop, summarize_phrase_scores


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_optional_tensor(path: str | None) -> torch.Tensor | None:
    if not path:
        return None
    return torch.load(path, map_location="cpu")


def _scores_from_token_table(row: dict[str, Any], token_scores: torch.Tensor | None, row_idx: int, topk_ratio: float) -> dict[str, float] | None:
    # This is a deterministic fallback for offline attribution tables. It does not
    # fake generation-time log-probs; it only converts supplied token evidence
    # scores into deletion/sufficiency-like score fields.
    scores = None
    if "token_scores" in row and isinstance(row["token_scores"], list):
        scores = torch.tensor(row["token_scores"], dtype=torch.float32)
    elif token_scores is not None and row_idx < token_scores.shape[0]:
        scores = token_scores[row_idx].float().flatten()
    if scores is None or scores.numel() == 0:
        return None
    k = max(1, int(round(scores.numel() * topk_ratio)))
    topk = torch.topk(scores, k=k).values
    original = float(scores.mean().item())
    evidence = float(topk.mean().item())
    masked = float((scores.sum() - topk.sum()).clamp_min(0).item() / max(scores.numel() - k, 1))
    random = float(scores[:: max(scores.numel() // k, 1)][:k].mean().item())
    return {
        "original_score": original,
        "topk_masked_score": masked,
        "evidence_only_score": evidence,
        "random_masked_score": random,
    }


def generate_phrase_score_rows(predictions: list[dict[str, Any]], token_scores: torch.Tensor | None = None, topk_ratio: float = 0.15) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows = []
    phrase_records = []
    for i, row in enumerate(predictions):
        rec = dict(row)
        text = rec.get("prediction") or rec.get("caption") or rec.get("text") or ""
        hits = [h.to_dict() for h in find_phrase_hits(text)]
        rec["phrase_hits"] = hits
        rec["phrase_hit_count"] = len(hits)
        generated = _scores_from_token_table(rec, token_scores, i, topk_ratio)
        if generated is not None and hits:
            rec.update(generated)
            rec["phrase_faithfulness"] = [dict(hit, **generated) for hit in hits]
            phrase_records.extend(rec["phrase_faithfulness"])
        out_rows.append(rec)
    return out_rows, {
        "count": len(out_rows),
        "with_phrase_hit": sum(1 for r in out_rows if r.get("phrase_hit_count", 0) > 0),
        "with_generated_scores": sum(1 for r in out_rows if "phrase_faithfulness" in r),
        **summarize_phrase_scores(phrase_records),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate phrase faithfulness score JSONL from ADAPT/FATE-X predictions and optional token attribution scores.")
    ap.add_argument("--predictions_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--token_scores_pt", default="")
    ap.add_argument("--topk_ratio", type=float, default=0.15)
    args = ap.parse_args()
    rows = _read_jsonl(args.predictions_jsonl)
    token_scores = _load_optional_tensor(args.token_scores_pt)
    out_rows, summary = generate_phrase_score_rows(rows, token_scores=token_scores, topk_ratio=args.topk_ratio)
    _write_jsonl(args.output_jsonl, out_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
