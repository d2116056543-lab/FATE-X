from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "experiment",
    "Narr_B4",
    "Narr_METEOR",
    "Narr_ROUGE",
    "Narr_CIDEr",
    "Reason_B4",
    "Reason_METEOR",
    "Reason_ROUGE",
    "Reason_CIDEr",
    "speed_RMSE",
    "course_RMSE",
    "phrase_deletion",
    "phrase_sufficiency",
    "avg_text_tokens",
    "avg_control_tokens",
    "latency",
    "peak_mem",
]


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig") or "{}")


def summarize_run(run_dir: Path) -> dict[str, Any]:
    epoch_dirs = sorted(p for p in run_dir.glob("epoch_*") if p.is_dir())
    epoch_dir = epoch_dirs[-1] if epoch_dirs else run_dir
    cap = _json(epoch_dir / "caption_metrics.json")
    ctrl = _json(epoch_dir / "control_metrics.json")
    faith = _json(epoch_dir / "phrase_faithfulness_summary.json")
    token_rows = []
    token_path = epoch_dir / "token_stats.jsonl"
    if token_path.exists():
        token_rows = [json.loads(line) for line in token_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    avg_text = None
    avg_control = None
    if token_rows:
        avg_text = sum(float(r.get("text_visual_tokens", 0)) for r in token_rows) / max(len(token_rows), 1)
        avg_control = sum(float(r.get("control_visual_tokens", 0)) for r in token_rows) / max(len(token_rows), 1)
    return {
        "experiment": run_dir.name,
        "Narr_B4": cap.get("Narr_B4"),
        "Narr_METEOR": cap.get("Narr_METEOR"),
        "Narr_ROUGE": cap.get("Narr_ROUGE"),
        "Narr_CIDEr": cap.get("Narr_CIDEr"),
        "Reason_B4": cap.get("Reason_B4"),
        "Reason_METEOR": cap.get("Reason_METEOR"),
        "Reason_ROUGE": cap.get("Reason_ROUGE"),
        "Reason_CIDEr": cap.get("Reason_CIDEr"),
        "speed_RMSE": ctrl.get("speed_RMSE"),
        "course_RMSE": ctrl.get("course_RMSE"),
        "phrase_deletion": faith.get("phrase_deletion"),
        "phrase_sufficiency": faith.get("phrase_sufficiency"),
        "avg_text_tokens": avg_text,
        "avg_control_tokens": avg_control,
        "latency": cap.get("latency") or ctrl.get("latency"),
        "peak_mem": cap.get("peak_mem") or ctrl.get("peak_mem"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FATE-X ablation CSV from output directories.")
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()
    rows = [summarize_run(Path(p)) for p in args.run_dirs]
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "fate_x_ablation_table", "rows": len(rows), "output_csv": str(out)}), flush=True)


if __name__ == "__main__":
    main()
