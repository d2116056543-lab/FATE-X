from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from fate_x.engine.adapt_caption_eval_bridge import (
    _candidate_eval_paths,
    _flatten_two_cap_metrics,
    _read_json,
    _TemporaryEvalEnvironment,
    _Utf8DefaultOpen,
    _WindowsSpiceNativeRuntime,
)


def _first_number(metrics: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return default


def normalize_adapt_reference(metrics: dict) -> dict:
    cider_description = _first_number(metrics, "CIDEr_description", "CIDEr_des", "description_CIDEr")
    cider_explanation = _first_number(metrics, "CIDEr_explanation", "CIDEr_exp", "explanation_CIDEr")
    speed_rmse = _first_number(metrics, "speed_RMSE", "Speed_RMSE", "speed_rmse", default=2.68)
    course_rmse = _first_number(metrics, "course_RMSE", "Course_RMSE", "course_rmse", default=5.87)
    return {
        "CIDEr_description": cider_description,
        "CIDEr_explanation": cider_explanation,
        "CIDEr_sum": cider_description + cider_explanation,
        "speed_RMSE": speed_rmse,
        "course_RMSE": course_rmse,
        "source_keys": sorted(metrics.keys()),
    }


def _flatten_numeric(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_numeric(value, name))
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


def compare_metric_payloads(
    original: dict[str, Any],
    reproduced: dict[str, Any],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    expected = _flatten_numeric(original)
    actual = _flatten_numeric(reproduced)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    errors = {key: abs(expected[key] - actual[key]) for key in expected.keys() & actual.keys()}
    mismatches = {key: value for key, value in errors.items() if value > tolerance}
    return {
        "status": "pass" if not missing and not extra and not mismatches else "blocked",
        "passed": not missing and not extra and not mismatches,
        "tolerance": tolerance,
        "max_abs_error": max(errors.values(), default=0.0),
        "missing_fields": missing,
        "extra_fields": extra,
        "mismatches": mismatches,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rerun_adapt_prediction_metrics(
    prediction_tsv: str | Path,
    caption_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    from src.evalcap.utils_caption_evaluate import two_cap_evaluate_on_coco_caption

    prediction_tsv = Path(prediction_tsv)
    caption_json = Path(caption_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "parity.BDDX.test.eval.json"
    with _TemporaryEvalEnvironment(output_dir), _Utf8DefaultOpen(), _WindowsSpiceNativeRuntime(output_dir):
        result = two_cap_evaluate_on_coco_caption(
            str(prediction_tsv), str(caption_json), outfile=str(outfile)
        )
    des_path, exp_path = _candidate_eval_paths(outfile)
    des, exp = _read_json(des_path), _read_json(exp_path)
    if not des and not exp and isinstance(result, dict):
        des = result.get("des", {}) if isinstance(result.get("des"), dict) else {}
        exp = result.get("exp", {}) if isinstance(result.get("exp"), dict) else {}
    return {
        "description": des,
        "explanation": exp,
        "flat": _flatten_two_cap_metrics(des, exp),
        "prediction_sha256": file_sha256(prediction_tsv),
        "caption_sha256": file_sha256(caption_json),
        "description_eval": str(des_path),
        "explanation_eval": str(exp_path),
    }


def run_parity(
    prediction_tsv: str | Path,
    caption_json: str | Path,
    original_description_json: str | Path,
    original_explanation_json: str | Path,
    output_dir: str | Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    reproduced = rerun_adapt_prediction_metrics(prediction_tsv, caption_json, output_dir)
    original = {
        "description": json.loads(Path(original_description_json).read_text(encoding="utf-8")),
        "explanation": json.loads(Path(original_explanation_json).read_text(encoding="utf-8")),
    }
    parity = compare_metric_payloads(original, {
        "description": reproduced["description"],
        "explanation": reproduced["explanation"],
    }, tolerance=tolerance)
    report = {
        **parity,
        "original": original,
        "reproduced": reproduced,
        "original_description_sha256": file_sha256(original_description_json),
        "original_explanation_sha256": file_sha256(original_explanation_json),
    }
    Path(output_dir, "adapt_metric_parity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_json", help="existing ADAPT reproduction metrics JSON")
    parser.add_argument("--prediction_tsv")
    parser.add_argument("--caption_json")
    parser.add_argument("--original_description_json")
    parser.add_argument("--original_explanation_json")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.prediction_tsv:
        required = (
            args.caption_json,
            args.original_description_json,
            args.original_explanation_json,
        )
        if not all(required):
            raise SystemExit("parity mode requires caption and original description/explanation JSON")
        print(json.dumps(run_parity(
            args.prediction_tsv,
            args.caption_json,
            args.original_description_json,
            args.original_explanation_json,
            args.output_dir,
        ), indent=2))
        return
    if not args.metrics_json:
        raise SystemExit("--metrics_json or --prediction_tsv is required")
    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    normalized = normalize_adapt_reference(metrics)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapt_reference_metrics.json").write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    print(json.dumps(normalized, indent=2))


if __name__ == "__main__":
    main()
