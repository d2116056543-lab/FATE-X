"""Build ADAPT-style captions_BDDX.json from the public BDD-X CSV.

The ADAPT preprocessing scripts expect a JSON object with parallel
``videos`` and ``annotations`` lists. The public BDD-X annotation release is a
CSV where each row contains up to 15 action/justification time segments for one
video. This script flattens those segments in CSV order and preserves ADAPT's
split-threshold assumption used in ``src/prepro/create_image_frame_tsv.py`` and
``src/prepro/tsv_preproc_BDDX.py``:

    training   annotations[:21143]
    validation annotations[21143:23662]
    testing    annotations[23662:]

It does not change labels or split thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlparse


def _video_stem(url_or_name: str) -> str:
    name = Path(urlparse(url_or_name).path).name or Path(url_or_name).name
    return Path(name).stem


def build(csv_path: Path) -> dict:
    videos = []
    annotations = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_name = _video_stem((row.get("Input.Video") or "").strip())
            if not vid_name:
                continue
            for idx in range(1, 16):
                action = (row.get(f"Answer.{idx}action") or "").strip()
                justification = (row.get(f"Answer.{idx}justification") or "").strip()
                start = (row.get(f"Answer.{idx}start") or "").strip()
                end = (row.get(f"Answer.{idx}end") or "").strip()
                if not action:
                    continue
                ann_id = len(annotations)
                item = {
                    "id": ann_id,
                    "vidName": vid_name,
                    "sTime": int(float(start)) if start else 0,
                    "eTime": int(float(end)) if end else 0,
                    "action": action,
                    "justification": justification,
                    "source_csv_row_video": (row.get("Input.Video") or "").strip(),
                    "source_answer_index": idx,
                }
                annotations.append(item)
                videos.append({"video_name": vid_name})
    return {"videos": videos, "annotations": annotations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = build(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    summary = {
        "output": str(args.output),
        "videos": len(data["videos"]),
        "annotations": len(data["annotations"]),
        "training_threshold_count": 21143,
        "validation_threshold_count": 23662 - 21143,
        "testing_threshold_count": max(0, len(data["annotations"]) - 23662),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
