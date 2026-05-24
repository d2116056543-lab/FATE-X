from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate_x.explain.phrase_attribution import find_phrase_hits
from fate_x.explain.phrase_counterfactual import summarize_phrase_scores


def _phrase_records_from_score_fields(rec: dict, hits: list[dict]) -> list[dict]:
    records = []
    score_table = rec.get("phrase_faithfulness") or rec.get("phrase_scores") or None
    if isinstance(score_table, list):
        for row in score_table:
            if isinstance(row, dict):
                records.append(row)
        return records
    for hit in hits:
        row = dict(hit)
        concept = hit.get("concept")
        if isinstance(score_table, dict) and concept in score_table and isinstance(score_table[concept], dict):
            row.update(score_table[concept])
        else:
            for field in ["original_score", "topk_masked_score", "evidence_only_score", "random_masked_score"]:
                if field in rec:
                    row[field] = rec[field]
        records.append(row)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate BDD-X phrase lexical hits and optional deletion/sufficiency faithfulness scores.")
    ap.add_argument("--predictions_jsonl", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = []
    phrase_records = []
    for line in Path(args.predictions_jsonl).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = rec.get("prediction") or rec.get("caption") or rec.get("text") or ""
        hits = [h.__dict__ for h in find_phrase_hits(text)]
        rec["phrase_hits"] = hits
        rec["phrase_hit_count"] = len(hits)
        rec_phrase_records = _phrase_records_from_score_fields(rec, hits)
        phrase_records.extend(rec_phrase_records)
        if rec_phrase_records:
            rec["phrase_faithfulness_summary"] = summarize_phrase_scores(rec_phrase_records)
        rows.append(rec)
    summary = {
        "count": len(rows),
        "with_phrase_hit": sum(1 for r in rows if r["phrase_hit_count"] > 0),
        "lexical_only_rows": sum(1 for r in rows if r.get("phrase_hit_count", 0) > 0 and not r.get("phrase_faithfulness_summary", {}).get("faithfulness_available", False)),
        **summarize_phrase_scores(phrase_records),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()