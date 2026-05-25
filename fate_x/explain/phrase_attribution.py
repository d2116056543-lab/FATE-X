from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PHRASE_ONTOLOGY: dict[str, dict[str, Any]] = {
    "traffic_light": {
        "priority": 100,
        "include": [
            r"\btraffic\s+light(s)?\b",
            r"\b(red|green|yellow)\s+light(s)?\b",
            r"\blight\s+(turns|turned|is)\s+(red|green|yellow)\b",
        ],
        "exclude": [r"\blight\s+traffic\b", r"\bstreet\s+light\b"],
    },
    "pedestrian": {
        "priority": 90,
        "include": [r"\bpedestrian(s)?\b", r"\bperson\b", r"\bpeople\s+(crossing|walking)\b"],
        "exclude": [],
    },
    "front_vehicle": {
        "priority": 80,
        "include": [
            r"\b(car|vehicle|truck|bus|cars|vehicles)\s+(ahead|in front|stopped ahead)\b",
            r"\bstopped\s+(car|vehicle|truck|bus|cars|vehicles)\b",
            r"\b(cars|vehicles)\s+are\s+stopped\s+ahead\b",
        ],
        "exclude": [],
    },
    "car_vehicle": {
        "priority": 40,
        "include": [r"\bcar(s)?\b", r"\bvehicle(s)?\b", r"\btruck(s)?\b", r"\bbus(es)?\b"],
        "exclude": [],
    },
    "lane_turn": {
        "priority": 70,
        "include": [r"\bturn(s|ing)?\s+(left|right)\b", r"\b(left|right)\s+lane\b", r"\bmerge(s|ing)?\b"],
        "exclude": [],
    },
    "clear_road": {
        "priority": 60,
        "include": [r"\btraffic\s+is\s+clear\b", r"\broad\s+is\s+clear\b", r"\bno\s+(cars|vehicles|traffic)\b"],
        "exclude": [],
    },
    "stop_sign": {"priority": 75, "include": [r"\bstop\s+sign(s)?\b"], "exclude": []},
    "obstacle": {"priority": 50, "include": [r"\bobstacle(s)?\b", r"\bblocking\b", r"\bparked\b"], "exclude": []},
}


@dataclass(frozen=True)
class PhraseHit:
    concept: str
    matched_text: str
    char_span: tuple[int, int]
    source_segment: str = "full"
    token_span: tuple[int, int] | None = None
    priority: int = 0

    @property
    def phrase(self) -> str:
        return self.matched_text

    @property
    def start(self) -> int:
        return self.char_span[0]

    @property
    def end(self) -> int:
        return self.char_span[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "phrase": self.phrase,
            "matched_text": self.matched_text,
            "start": self.start,
            "end": self.end,
            "char_span": list(self.char_span),
            "token_span": list(self.token_span) if self.token_span is not None else None,
            "source_segment": self.source_segment,
            "priority": self.priority,
        }


def _compile_rule(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


def _excluded(text: str, start: int, end: int, excludes: list[str]) -> bool:
    local = text[max(0, start - 12) : min(len(text), end + 12)]
    return any(_compile_rule(p).search(local) for p in excludes)


def load_phrase_ontology(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    if not path:
        return DEFAULT_PHRASE_ONTOLOGY
    p = Path(path)
    if not p.exists():
        return DEFAULT_PHRASE_ONTOLOGY
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(p.read_text(encoding="utf-8-sig")) or {}
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return DEFAULT_PHRASE_ONTOLOGY


def _segment_text(record_or_text: str | dict[str, Any], justification_only: bool) -> tuple[str, str, int]:
    if isinstance(record_or_text, str):
        return record_or_text, "full", 0
    if justification_only:
        for key in ("justification", "reason", "explanation"):
            value = record_or_text.get(key)
            if isinstance(value, str) and value.strip():
                return value, "justification", 0
    for key in ("prediction", "caption", "text"):
        value = record_or_text.get(key)
        if isinstance(value, str):
            return value, "full", 0
    return "", "full", 0


def _token_span_from_offsets(char_span: tuple[int, int], offsets: list[tuple[int, int]] | None) -> tuple[int, int] | None:
    if not offsets:
        return None
    start, end = char_span
    token_ids = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    if not token_ids:
        return None
    return min(token_ids), max(token_ids) + 1


def find_phrase_hits(
    text: str | dict[str, Any],
    vocab: dict[str, list[str]] | None = None,
    *,
    ontology: dict[str, dict[str, Any]] | None = None,
    justification_only: bool = False,
    token_offsets: list[tuple[int, int]] | None = None,
) -> list[PhraseHit]:
    if vocab is not None and ontology is None:
        ontology = {
            concept: {"priority": 0, "include": [re.escape(p) for p in phrases], "exclude": []}
            for concept, phrases in vocab.items()
        }
    ontology = ontology or DEFAULT_PHRASE_ONTOLOGY
    segment_text, source_segment, base_offset = _segment_text(text, justification_only)
    candidates: list[PhraseHit] = []
    for concept, spec in sorted(ontology.items(), key=lambda kv: int(kv[1].get("priority", 0)), reverse=True):
        priority = int(spec.get("priority", 0))
        excludes = [str(x) for x in spec.get("exclude", []) or []]
        for pattern in spec.get("include", []) or []:
            for match in _compile_rule(str(pattern)).finditer(segment_text):
                start = base_offset + match.start()
                end = base_offset + match.end()
                if _excluded(segment_text, match.start(), match.end(), excludes):
                    continue
                candidates.append(
                    PhraseHit(
                        concept=concept,
                        matched_text=match.group(0),
                        char_span=(start, end),
                        source_segment=source_segment,
                        token_span=_token_span_from_offsets((start, end), token_offsets),
                        priority=priority,
                    )
                )
    # Priority suppresses lower-priority overlapping concepts.
    chosen: list[PhraseHit] = []
    occupied: list[tuple[int, int]] = []
    for hit in sorted(candidates, key=lambda h: (-h.priority, h.start, h.end, h.concept)):
        if any(not (hit.end <= a or hit.start >= b) for a, b in occupied):
            continue
        chosen.append(hit)
        occupied.append(hit.char_span)
    chosen.sort(key=lambda h: (h.start, h.end, -h.priority, h.concept))
    return chosen


def phrase_token_mask(text: str, tokenizer, vocab: dict[str, list[str]] | None = None):
    encoded = tokenizer(text, return_offsets_mapping=True) if tokenizer is not None else None
    offsets = None
    if encoded is not None:
        offsets = [tuple(x) for x in encoded.get("offset_mapping", [])]
    hits = find_phrase_hits(text, vocab, token_offsets=offsets)
    return {"hits": hits, "encoded": encoded}
