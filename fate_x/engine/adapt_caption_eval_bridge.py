from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


TEXT_METRIC_NAMES = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE")
_ADAPT_SEGMENT_SUFFIX_RE = re.compile(r":\d+$")


def _cap_list(text: str, conf: float = 1.0) -> str:
    return json.dumps([{"caption": text, "conf": float(conf)}], ensure_ascii=False)


def normalize_adapt_image_id(img_key: Any) -> str:
    """Map ADAPT segment keys back to COCO caption image ids."""
    return _ADAPT_SEGMENT_SUFFIX_RE.sub("", str(img_key))


def write_adapt_sep_caption_tsv(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    """Write ADAPT-compatible two-caption TSV: img_key, description json, explanation json."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            key = normalize_adapt_image_id(row["img_key"])
            des = _cap_list(str(row.get("description", "")), float(row.get("description_conf", 1.0)))
            exp = _cap_list(str(row.get("explanation", "")), float(row.get("explanation_conf", 1.0)))
            f.write(f"{key}\t{des}\t{exp}\n")
    return output_path


def caption_file_from_loader(loader: Any) -> str | None:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    if hasattr(dataset, "get_caption_file_in_coco_format"):
        path = dataset.get_caption_file_in_coco_format()
        return str(path) if path else None
    return getattr(dataset, "caption_file", None)


def _read_json(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}


def _candidate_eval_paths(outfile: Path) -> tuple[Path, Path]:
    text = str(outfile)
    if "BDDX" in text:
        return Path(text.replace("BDDX", "BDDX_des")), Path(text.replace("BDDX", "BDDX_exp"))
    return outfile.with_name(outfile.stem + "_des" + outfile.suffix), outfile.with_name(outfile.stem + "_exp" + outfile.suffix)


def _flatten_two_cap_metrics(des: dict[str, float], exp: dict[str, float]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in TEXT_METRIC_NAMES:
        if name in des:
            metrics[f"{name}_des"] = float(des[name])
        if name in exp:
            metrics[f"{name}_exp"] = float(exp[name])
    if "CIDEr_des" in metrics and "CIDEr_exp" in metrics:
        metrics["CIDEr_des+exp"] = metrics["CIDEr_des"] + metrics["CIDEr_exp"]
    return metrics


class _TemporaryEvalEnvironment:
    """Keep ADAPT caption evaluation on the run drive instead of the small system drive."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._old: dict[str, str | None] = {}

    def __enter__(self) -> None:
        tmp_dir = self.output_dir / "adapt_eval_runtime_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        updates = {
            "TMP": str(tmp_dir),
            "TEMP": str(tmp_dir),
            "SPICE_DISABLE_CACHE": "1",
        }
        java_tmp_opt = f"-Djava.io.tmpdir={tmp_dir}"
        current_java_opts = os.environ.get("JAVA_TOOL_OPTIONS", "")
        if java_tmp_opt not in current_java_opts:
            updates["JAVA_TOOL_OPTIONS"] = (current_java_opts + " " + java_tmp_opt).strip()
        for key, value in updates.items():
            self._old[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_adapt_sep_caption_eval(
    prediction_rows: Iterable[dict[str, Any]],
    loader: Any,
    output_dir: str | Path,
    prefix: str = "pred.BDDX.test.beam1.max30",
) -> dict[str, Any]:
    """Run ADAPT's official two-caption evaluator when the reference tooling is installed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predict_file = write_adapt_sep_caption_tsv(prediction_rows, output_dir / f"{prefix}.tsv")
    caption_file = caption_file_from_loader(loader)
    if not caption_file:
        return {
            "text_metrics_available": False,
            "text_metrics_blocker": "loader dataset does not expose get_caption_file_in_coco_format/caption_file",
            "adapt_predict_file": str(predict_file),
        }
    try:
        from src.evalcap.utils_caption_evaluate import two_cap_evaluate_on_coco_caption
    except Exception as exc:
        return {
            "text_metrics_available": False,
            "text_metrics_blocker": f"ADAPT src.evalcap evaluator import failed: {exc}",
            "adapt_predict_file": str(predict_file),
            "adapt_caption_file": str(caption_file),
        }
    outfile = predict_file.with_suffix(".eval.json")
    try:
        with _TemporaryEvalEnvironment(output_dir):
            result = two_cap_evaluate_on_coco_caption(str(predict_file), str(caption_file), outfile=str(outfile))
    except Exception as exc:
        return {
            "text_metrics_available": False,
            "text_metrics_blocker": f"ADAPT two_cap_evaluate_on_coco_caption failed: {exc}",
            "adapt_predict_file": str(predict_file),
            "adapt_caption_file": str(caption_file),
        }

    des_path, exp_path = _candidate_eval_paths(outfile)
    des = _read_json(des_path)
    exp = _read_json(exp_path)
    if not des and not exp and isinstance(result, dict):
        # Some evaluator versions return a pair-like payload instead of writing split files.
        des = {k: float(v) for k, v in result.get("des", {}).items()} if isinstance(result.get("des"), dict) else {}
        exp = {k: float(v) for k, v in result.get("exp", {}).items()} if isinstance(result.get("exp"), dict) else {}
    metrics = _flatten_two_cap_metrics(des, exp)
    if not metrics:
        return {
            "text_metrics_available": False,
            "text_metrics_blocker": "ADAPT evaluator ran but no split description/explanation metric JSON was found",
            "adapt_predict_file": str(predict_file),
            "adapt_caption_file": str(caption_file),
            "adapt_eval_file": str(outfile),
        }
    metrics.update(
        {
            "text_metrics_available": True,
            "adapt_predict_file": str(predict_file),
            "adapt_caption_file": str(caption_file),
            "adapt_eval_file": str(outfile),
        }
    )
    return metrics
