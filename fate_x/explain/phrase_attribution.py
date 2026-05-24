from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_PHRASE_VOCAB = {
    "traffic_light": ["traffic light", "light", "red light", "green light", "light turns"],
    "stop_sign": ["stop sign"],
    "car_vehicle": ["car", "vehicle", "truck", "bus", "traffic", "cars"],
    "pedestrian": ["pedestrian", "person", "people", "crossing"],
    "left": ["left", "left lane", "turning left"],
    "right": ["right", "right lane", "turning right"],
    "clear": ["clear", "no traffic", "empty"],
    "obstacle": ["obstacle", "blocking", "parked", "around"],
}


@dataclass(frozen=True)
class PhraseHit:
    concept: str
    phrase: str
    start: int
    end: int


def find_phrase_hits(text: str, vocab: dict[str, list[str]] | None = None) -> list[PhraseHit]:
    vocab = vocab or DEFAULT_PHRASE_VOCAB
    lower = text.lower()
    hits: list[PhraseHit] = []
    for concept, phrases in vocab.items():
        for phrase in phrases:
            for m in re.finditer(re.escape(phrase.lower()), lower):
                hits.append(PhraseHit(concept, phrase, m.start(), m.end()))
    hits.sort(key=lambda x: (x.start, x.end, x.concept))
    return hits


def phrase_token_mask(text: str, tokenizer, vocab: dict[str, list[str]] | None = None):
    # Lightweight helper: returns phrase character hits and tokenization if tokenizer is supplied.
    hits = find_phrase_hits(text, vocab)
    encoded = tokenizer(text, return_offsets_mapping=True) if tokenizer is not None else None
    return {"hits": hits, "encoded": encoded}
