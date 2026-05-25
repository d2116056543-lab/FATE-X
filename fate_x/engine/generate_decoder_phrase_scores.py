from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fate_x.explain.phrase_attribution import find_phrase_hits
from fate_x.engine.eval_phrase_faithfulness import summarize_rows


def _token_offsets_from_tokens(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    lower = text.lower()
    for token in tokens:
        clean = str(token).replace("##", "")
        idx = lower.find(clean.lower(), cursor)
        if idx < 0:
            idx = cursor
        end = idx + len(clean)
        offsets.append((idx, end))
        cursor = end
    return offsets


def _span_token_indices(hit_start: int, hit_end: int, offsets: list[tuple[int, int]]) -> list[int]:
    out = []
    for i, (start, end) in enumerate(offsets):
        if end <= hit_start or start >= hit_end:
            continue
        out.append(i)
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _score_span(logprobs: list[float], indices: list[int]) -> float:
    usable = [float(logprobs[i]) for i in indices if 0 <= i < len(logprobs)]
    return _mean(usable) if usable else float("nan")


def score_decoder_phrase_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach phrase faithfulness scores from decoder token log-prob fields.

    Required row fields:
      prediction/caption/text, tokens, token_logprobs

    Optional perturbation fields:
      topk_masked_token_logprobs, evidence_only_token_logprobs,
      random_masked_token_logprobs
    """
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("prediction") or row.get("caption") or row.get("text") or "")
        tokens = [str(x) for x in row.get("tokens", [])]
        token_logprobs = row.get("token_logprobs")
        if not text or not tokens or token_logprobs is None:
            raise ValueError("decoder phrase scoring requires text, tokens, and token_logprobs")
        offsets = row.get("token_offsets")
        if offsets is None:
            offsets = _token_offsets_from_tokens(text, tokens)
        else:
            offsets = [(int(x[0]), int(x[1])) for x in offsets]
        phrase_items = []
        for hit in find_phrase_hits(text):
            indices = _span_token_indices(hit.start, hit.end, offsets)
            if not indices:
                continue
            item = {
                "concept": hit.concept,
                "phrase": hit.phrase,
                "start": hit.start,
                "end": hit.end,
                "token_indices": indices,
                "original_score": _score_span(token_logprobs, indices),
            }
            if "topk_masked_token_logprobs" in row:
                item["topk_masked_score"] = _score_span(row["topk_masked_token_logprobs"], indices)
            if "evidence_only_token_logprobs" in row:
                item["evidence_only_score"] = _score_span(row["evidence_only_token_logprobs"], indices)
            if "random_masked_token_logprobs" in row:
                item["random_masked_score"] = _score_span(row["random_masked_token_logprobs"], indices)
            phrase_items.append(item)
        out = dict(row)
        out["phrase_hits"] = [
            {"concept": x["concept"], "phrase": x["phrase"], "start": x["start"], "end": x["end"]}
            for x in phrase_items
        ]
        out["phrase_hit_count"] = len(phrase_items)
        out["phrase_faithfulness"] = phrase_items
        scored_rows.append(out)
    return scored_rows, summarize_rows(scored_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate phrase faithfulness rows from decoder token log-probs.")
    ap.add_argument("--decoder_jsonl", required=True, help="JSONL with generated text, tokens and decoder log-probs.")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--summary_json", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.decoder_jsonl).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    scored, summary = score_decoder_phrase_rows(rows)
    with Path(args.output_jsonl).open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
