"""Preflight checks for the ADAPT raw BDD-X preprocessing route."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlparse


def stem_from_url(url: str) -> str:
    return Path(urlparse(url).path).stem


def load_required_videos(csv_path: Path) -> list[str]:
    names = []
    seen = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = stem_from_url((row.get("Input.Video") or "").strip())
            if stem and stem not in seen:
                seen.add(stem)
                names.append(stem)
    return names


def count_json_annotations(path: Path) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data.get("annotations", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--captions_json", required=True, type=Path)
    parser.add_argument("--raw_video_dir", required=True, type=Path)
    parser.add_argument("--frame_dir", required=True, type=Path)
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--sample_missing", type=int, default=25)
    args = parser.parse_args()

    required = load_required_videos(args.csv)
    present = []
    missing = []
    for stem in required:
        path = args.raw_video_dir / f"{stem}.mov"
        if path.exists() and path.stat().st_size > 0:
            present.append(stem)
        else:
            missing.append(stem)

    dataset_expected = [
        args.dataset_dir / "training_32frames.yaml",
        args.dataset_dir / "testing_32frames.yaml",
        args.dataset_dir / "training_32frames.img.tsv",
        args.dataset_dir / "testing_32frames.img.tsv",
    ]
    result = {
        "csv": str(args.csv),
        "captions_json": str(args.captions_json),
        "captions_json_exists": args.captions_json.exists(),
        "captions_json_annotations": count_json_annotations(args.captions_json),
        "raw_video_dir": str(args.raw_video_dir),
        "required_unique_videos": len(required),
        "present_unique_videos": len(present),
        "missing_unique_videos": len(missing),
        "missing_sample": missing[: args.sample_missing],
        "frame_dir": str(args.frame_dir),
        "frame_dir_exists": args.frame_dir.exists(),
        "dataset_dir": str(args.dataset_dir),
        "expected_processed_files": {
            str(path): path.exists() for path in dataset_expected
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(2)
    if not args.captions_json.exists():
        raise SystemExit(3)


if __name__ == "__main__":
    main()
