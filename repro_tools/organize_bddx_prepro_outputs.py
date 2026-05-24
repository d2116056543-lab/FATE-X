"""Organize ADAPT BDD-X raw-preprocessing outputs into expected 32-frame names."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


SPLITS = ("training", "validation", "testing")


def copy_with_index(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    for suffix in (".lineidx", ".lineidx.8b"):
        src_idx = Path(str(src).replace(".tsv", suffix))
        if src_idx.exists():
            dst_idx = Path(str(dst).replace(".tsv", suffix))
            shutil.copy2(src_idx, dst_idx)


def row_count(tsv: Path) -> int:
    if not tsv.exists():
        return 0
    with tsv.open("rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=Path)
    parser.add_argument("--frame_tsv_dir", required=True, type=Path)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLITS:
        prefix = f"{split}_{args.num_frames}frames"
        frame_src = args.frame_tsv_dir / (
            f"{split}_{args.num_frames}frames_img_size{args.image_size}.img.tsv"
        )
        frame_dst = args.dataset_dir / f"{prefix}.img.tsv"
        copy_with_index(frame_src, frame_dst)

        generated = {
            "label": args.dataset_dir / f"{split}.label.tsv",
            "caption": args.dataset_dir / f"{split}.caption.tsv",
            "caption_linelist": args.dataset_dir / f"{split}.caption.linelist.tsv",
            "caption_coco_format": args.dataset_dir / f"{split}.caption_coco_format.json",
        }
        final = {
            "label": args.dataset_dir / f"{prefix}.label.tsv",
            "caption": args.dataset_dir / f"{prefix}.caption.tsv",
            "caption_linelist": args.dataset_dir / f"{prefix}.caption.linelist.tsv",
            "caption_coco_format": args.dataset_dir / f"{prefix}.caption_coco_format.json",
        }
        for key, src in generated.items():
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, final[key])

        yaml_path = args.dataset_dir / f"{prefix}.yaml"
        yaml_data = {
            "img": frame_dst.name,
            "label": final["label"].name,
            "caption": final["caption"].name,
            "caption_linelist": final["caption_linelist"].name,
            "caption_coco_format": final["caption_coco_format"].name,
        }
        yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
        summary[split] = {
            "yaml": str(yaml_path),
            "img": str(frame_dst),
            "img_rows": row_count(frame_dst),
            "label_rows": row_count(final["label"]),
            "caption_rows": row_count(final["caption"]),
        }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
