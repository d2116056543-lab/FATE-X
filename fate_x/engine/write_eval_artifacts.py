from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _phrase_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"phrase_records": 0, "faithfulness_available": False}
    deletion = [float(r["deletion_score"]) for r in scores if r.get("deletion_score") is not None]
    suff = [float(r["sufficiency_score"]) for r in scores if r.get("sufficiency_score") is not None]
    return {
        "phrase_records": len(scores),
        "faithfulness_available": bool(deletion or suff),
        "phrase_deletion": sum(deletion) / max(len(deletion), 1) if deletion else None,
        "phrase_sufficiency": sum(suff) / max(len(suff), 1) if suff else None,
    }


def write_fate_x_eval_artifacts(
    output_dir: str | Path,
    epoch: int,
    rows: list[dict[str, Any]],
    *,
    caption_metrics: dict[str, Any] | None = None,
    control_metrics: dict[str, Any] | None = None,
    run_manifest: dict[str, Any] | None = None,
) -> Path:
    """Write the minimum diagnosable FATE-X eval artifact set."""
    epoch_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    caption_metrics = caption_metrics or {"available": False, "reason": "not computed in this smoke/eval wrapper"}
    control_metrics = control_metrics or {"available": False, "reason": "not computed in this smoke/eval wrapper"}
    _write_json(epoch_dir / "caption_metrics.json", caption_metrics)
    _write_json(epoch_dir / "control_metrics.json", control_metrics)

    predictions: list[dict[str, Any]] = []
    phrase_hits: list[dict[str, Any]] = []
    phrase_scores: list[dict[str, Any]] = []
    token_stats: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        pred = {
            "sample_id": row.get("sample_id") or row.get("id"),
            "split": row.get("split"),
            "gt_action": row.get("gt_action"),
            "gt_justification": row.get("gt_justification"),
            "pred_action": row.get("pred_action"),
            "pred_justification": row.get("pred_justification") or row.get("prediction"),
            "generated_tokens": row.get("generated_tokens") or row.get("tokens") or [],
            "token_logprobs": row.get("token_logprobs") or [],
            "phrase_hits": row.get("phrase_hits") or [],
            "token_stats": row.get("token_stats") or {},
        }
        predictions.append(pred)
        for hit in row.get("phrase_hits") or []:
            phrase_hits.append({"sample_id": pred["sample_id"], **hit})
        faithfulness = row.get("phrase_scores")
        if faithfulness is None:
            raw_faith = row.get("phrase_faithfulness")
            if isinstance(raw_faith, dict):
                faithfulness = raw_faith.get("phrases", [])
            elif isinstance(raw_faith, list):
                faithfulness = raw_faith
            else:
                faithfulness = []
        for score in faithfulness or []:
            phrase_scores.append({"sample_id": pred["sample_id"], **score})
        if row.get("token_stats"):
            token_stats.append({"sample_id": pred["sample_id"], **row["token_stats"]})
        if row.get("failure"):
            failures.append({"sample_id": pred["sample_id"], "failure": row["failure"]})

    _write_jsonl(epoch_dir / "predictions.jsonl", predictions)
    _write_jsonl(epoch_dir / "phrase_hits.jsonl", phrase_hits)
    _write_jsonl(epoch_dir / "phrase_scores.jsonl", phrase_scores)
    _write_jsonl(epoch_dir / "token_stats.jsonl", token_stats)
    _write_json(epoch_dir / "phrase_faithfulness_summary.json", _phrase_summary(phrase_scores))
    _write_jsonl(epoch_dir / "failure_cases.jsonl", failures)
    _write_json(epoch_dir / "run_manifest.json", run_manifest or {})
    return epoch_dir
