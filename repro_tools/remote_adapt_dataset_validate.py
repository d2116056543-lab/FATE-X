from pathlib import Path
import json
import os

ROOT = Path(r"E:\sbw\ADAPT_repro\ADAPT")
DATA_ROOT = ROOT / "ADAPT_PREPROCESSED_DATASET"

def line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)

def exists_size(path: Path):
    return {"exists": path.exists(), "size": path.stat().st_size if path.exists() else None}

def split_summary(base: Path):
    result = {"path": str(base), "exists": base.exists(), "splits": {}}
    for split in ["training", "validation", "testing"]:
        files = {}
        names = [
            f"{split}_32frames.yaml",
            f"{split}_32frames_caption_coco_format.json",
            f"{split}.label.tsv",
            f"{split}.caption.tsv",
            f"{split}.caption.linelist.tsv",
            f"{split}.linelist.tsv",
            f"{split}.img.tsv",
            f"frame_tsv/{split}_32frames_img_size256.img.tsv",
            f"frame_tsv_part/{split}_32frames_img_size256.img.tsv",
        ]
        for name in names:
            p = base / name
            if p.exists():
                item = exists_size(p)
                if p.suffix in [".tsv", ".json"] and p.stat().st_size < 200_000_000:
                    try:
                        item["lines"] = line_count(p)
                    except Exception as exc:
                        item["line_error"] = repr(exc)
                if p.suffix == ".json" and p.stat().st_size < 100_000_000:
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        if isinstance(data, dict):
                            item["json_keys"] = list(data.keys())[:8]
                            if "annotations" in data:
                                item["annotations"] = len(data["annotations"])
                            if "images" in data:
                                item["images"] = len(data["images"])
                    except Exception as exc:
                        item["json_error"] = repr(exc)
                files[name] = item
        result["splits"][split] = files
    return result

summary = {
    "download_root": str(DATA_ROOT),
    "download_root_exists": DATA_ROOT.exists(),
    "readme": (DATA_ROOT / "Readme.txt").read_text(encoding="utf-8", errors="ignore") if (DATA_ROOT / "Readme.txt").exists() else None,
    "dirs": {},
}
for rel in [
    "datasets/BDDX",
    "datasets/BDDX_des",
    "datasets/BDDX_exp",
    "datasets_part/BDDX",
    "datasets_part/BDDX_des",
    "datasets_part/BDDX_exp",
]:
    summary["dirs"][rel] = split_summary(DATA_ROOT / rel)

out = ROOT / "repro_logs" / "adapt_preprocessed_dataset_validation.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(out)
print(json.dumps({
    rel: {
        split: {
            name: {k: v for k, v in item.items() if k in ["exists", "size", "lines", "annotations", "images"]}
            for name, item in info["splits"][split].items()
        }
        for split in info["splits"]
    }
    for rel, info in summary["dirs"].items()
}, indent=2, ensure_ascii=False))
