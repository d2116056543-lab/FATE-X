from __future__ import annotations

from collections import Counter


def _ngrams(text: str, n: int) -> Counter:
    toks = text.lower().split()
    return Counter(tuple(toks[i:i+n]) for i in range(max(0, len(toks) - n + 1)))


def compute_text_cider_proxy(preds: list[str], refs: list[str]) -> float:
    scores = []
    for pred, ref in zip(preds, refs):
        per_n = []
        for n in range(1, 5):
            p = _ngrams(pred, n)
            r = _ngrams(ref, n)
            denom = sum(p.values()) + sum(r.values())
            if denom == 0:
                per_n.append(0.0)
            else:
                overlap = sum((p & r).values())
                per_n.append(2.0 * overlap / denom)
        scores.append(sum(per_n) / 4.0)
    return float(sum(scores) / max(1, len(scores)))
