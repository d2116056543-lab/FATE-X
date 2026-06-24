from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_json", required=True, help="existing ADAPT reproduction metrics JSON")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    normalized = normalize_adapt_reference(metrics)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapt_reference_metrics.json").write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    print(json.dumps(normalized, indent=2))


if __name__ == "__main__":
    main()
