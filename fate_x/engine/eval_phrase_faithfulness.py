from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate_x.explain.phrase_attribution import find_phrase_hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_jsonl", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = []
    for line in Path(args.predictions_jsonl).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = rec.get("prediction") or rec.get("caption") or rec.get("text") or ""
        hits = [h.__dict__ for h in find_phrase_hits(text)]
        rec["phrase_hits"] = hits
        rec["phrase_hit_count"] = len(hits)
        rows.append(rec)
    summary = {"count": len(rows), "with_phrase_hit": sum(1 for r in rows if r["phrase_hit_count"] > 0)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
