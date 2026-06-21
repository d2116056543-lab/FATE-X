from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


SPEED_LABELS = ["stop", "slow", "maintain", "accelerate"]
DIRECTION_LABELS = ["left", "straight", "right"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s-]", " ", str(text).lower())).strip()


def classify_speed_action(text: str) -> str | None:
    t = _normalize(text)
    if not t:
        return None
    if any(p in t for p in ("stop", "stopped", "stopping", "red light")):
        return "stop"
    if any(p in t for p in ("slow", "slowing", "slows", "brake", "braking", "decelerate", "decelerating", "carefully moving")):
        return "slow"
    if any(p in t for p in ("accelerate", "accelerates", "accelerating", "speed up", "speeds up")):
        return "accelerate"
    if any(p in t for p in ("steady", "constant", "continues", "continue", "moving", "driving", "goes", "going", "forward")):
        return "maintain"
    return None


def classify_direction_action(text: str) -> str | None:
    t = _normalize(text)
    if not t:
        return None
    if "left" in t:
        return "left"
    if "right" in t:
        return "right"
    if any(p in t for p in ("straight", "forward", "ahead")):
        return "straight"
    return None


def _read_gt_actions(caption_tsv: str | Path) -> dict[str, str]:
    actions: dict[str, str] = {}
    with Path(caption_tsv).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            key = parts[0]
            try:
                rows = json.loads(parts[1])
            except json.JSONDecodeError:
                continue
            if rows:
                actions[key] = str(rows[0].get("action") or rows[0].get("caption") or "")
    return actions


def _read_pred_actions(pred_tsv: str | Path) -> dict[str, str]:
    actions: dict[str, str] = {}
    with Path(pred_tsv).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            key = parts[0]
            try:
                rows = json.loads(parts[1])
            except json.JSONDecodeError:
                continue
            if rows:
                actions[key] = str(rows[0].get("caption") or rows[0].get("action") or "")
    return actions


def _classification_report(
    gt: dict[str, str],
    pred: dict[str, str],
    *,
    labels: list[str],
    classifier,
) -> dict[str, Any]:
    confusion = {g: {p: 0 for p in labels} for g in labels}
    support = Counter()
    predicted = Counter()
    unknown_gt = 0
    unknown_pred = 0
    correct = 0
    evaluated = 0
    for key, gt_text in gt.items():
        gt_label = classifier(gt_text)
        pred_label = classifier(pred.get(key, ""))
        if gt_label is None:
            unknown_gt += 1
            continue
        if pred_label is None:
            unknown_pred += 1
            continue
        if gt_label not in labels or pred_label not in labels:
            continue
        support[gt_label] += 1
        predicted[pred_label] += 1
        confusion[gt_label][pred_label] += 1
        correct += int(gt_label == pred_label)
        evaluated += 1

    per_class: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) else None
        if f1 is not None and math.isfinite(f1):
            f1_values.append(f1)
        if recall is not None and math.isfinite(recall):
            recall_values.append(recall)
        per_class[label] = {
            "support": int(support[label]),
            "predicted": int(predicted[label]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "labels": labels,
        "evaluated_count": int(evaluated),
        "accuracy": correct / evaluated if evaluated else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "macro_recall": sum(recall_values) / len(recall_values) if recall_values else None,
        "unknown_gt_count": int(unknown_gt),
        "unknown_pred_count": int(unknown_pred),
        "per_class": per_class,
        "confusion": confusion,
    }


def evaluate_action_text_decisions(pred_tsv: str | Path, caption_tsv: str | Path) -> dict[str, Any]:
    gt = _read_gt_actions(caption_tsv)
    pred = _read_pred_actions(pred_tsv)
    matched = sorted(set(gt) & set(pred))
    gt_matched = {k: gt[k] for k in matched}
    pred_matched = {k: pred[k] for k in matched}
    return {
        "metric_family": "bddx_action_text_decision_proxy",
        "pred_tsv": str(pred_tsv),
        "caption_tsv": str(caption_tsv),
        "gt_count": len(gt),
        "pred_count": len(pred),
        "matched_count": len(matched),
        "speed_decision": _classification_report(gt_matched, pred_matched, labels=SPEED_LABELS, classifier=classify_speed_action),
        "direction_decision": _classification_report(gt_matched, pred_matched, labels=DIRECTION_LABELS, classifier=classify_direction_action),
    }
