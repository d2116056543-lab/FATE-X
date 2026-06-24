from __future__ import annotations

import builtins
import json
import os
import re
import zipfile
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



class _WindowsSpiceNativeRuntime:
    """Make ADAPT SPICE load its bundled LMDB JNI library on Windows or Linux."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def __enter__(self) -> None:
        self._old_path = os.environ.get("PATH")
        self._spice_module = None
        self._old_check_call = None
        self._old_cache_dir = None
        self._old_temp_dir = None
        try:
            from src.evalcap.coco_caption.pycocoevalcap.spice import spice as spice_module
        except Exception:
            return
        spice_dir = Path(spice_module.__file__).resolve().parent
        lib_dir = spice_dir / "lib"
        self._prepare_lmdb_native(lib_dir)
        os.environ["PATH"] = f"{lib_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        self._spice_module = spice_module
        self._old_cache_dir = getattr(spice_module, "CACHE_DIR", None)
        self._old_temp_dir = getattr(spice_module, "TEMP_DIR", None)
        spice_module.CACHE_DIR = str(self._cache_dir())
        spice_module.TEMP_DIR = str(self.output_dir / "spice_tmp")
        Path(spice_module.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(spice_module.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        self._old_check_call = spice_module.subprocess.check_call

        def _check_call(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            patched_cmd = self._patch_spice_java_command(cmd, lib_dir)
            return self._old_check_call(patched_cmd, *args, **kwargs)

        spice_module.subprocess.check_call = _check_call

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._spice_module is not None and self._old_check_call is not None:
            self._spice_module.subprocess.check_call = self._old_check_call
        if self._spice_module is not None:
            if self._old_cache_dir is not None:
                self._spice_module.CACHE_DIR = self._old_cache_dir
            if self._old_temp_dir is not None:
                self._spice_module.TEMP_DIR = self._old_temp_dir
        if self._old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = self._old_path

    def _cache_dir(self) -> Path:
        override = os.environ.get("ACPR_DYNFLOW_SPICE_CACHE_ROOT")
        if override:
            return Path(override) / "cache"
        g_root = Path("G:/sbw/FATE_Drive/acpr_dynflow_spice_cache")
        if g_root.anchor and g_root.exists():
            return g_root / "cache"
        return self.output_dir / "spice_cache"

    @staticmethod
    def _prepare_lmdb_native(lib_dir: Path) -> None:
        if os.name == "nt":
            jar_path = lib_dir / "lmdbjni-win64-0.4.6.jar"
            archive_member = "META-INF/native/windows64/lmdbjni.dll"
            names = ("lmdbjni.dll", "lmdbjni64-0.4.6.dll", "lmdbjni-0.4.6.dll")
        else:
            jar_path = lib_dir / "lmdbjni-linux64-0.4.6.jar"
            archive_member = "META-INF/native/linux64/liblmdbjni.so"
            names = ("liblmdbjni.so", "liblmdbjni64-0.4.6.so", "liblmdbjni-0.4.6.so")
        if not jar_path.exists():
            return
        existing = [lib_dir / name for name in names]
        if all(path.exists() for path in existing):
            return
        with zipfile.ZipFile(jar_path) as jar:
            data = jar.read(archive_member)
        for path in existing:
            if not path.exists():
                path.write_bytes(data)

    @staticmethod
    def _patch_spice_java_command(cmd: Any, lib_dir: Path) -> Any:
        if not isinstance(cmd, (list, tuple)) or not cmd:
            return cmd
        parts = [str(part) for part in cmd]
        if parts[0].lower() != "java" or "spice-1.0.jar" not in parts:
            return cmd
        # ADAPT's bundled spice.py uses `java -jar -Xmx8G spice-1.0.jar`; make it legal
        # before injecting the Windows native-library search path.
        if len(parts) >= 4 and parts[1] == "-jar" and parts[2].startswith("-Xmx"):
            parts = [parts[0], parts[2], "-jar", parts[3], *parts[4:]]
        lib_arg = f"-Djava.library.path={lib_dir}"
        if lib_arg not in parts:
            parts.insert(1, lib_arg)
        return parts

class _Utf8DefaultOpen:
    """Make ADAPT evaluator text reads deterministic on non-UTF8 Windows locales."""

    def __enter__(self) -> None:
        self._old_open = builtins.open

        def _open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            text_mode = "b" not in mode
            if text_mode and "encoding" not in kwargs:
                kwargs["encoding"] = "utf-8"
            return self._old_open(file, mode, *args, **kwargs)

        builtins.open = _open

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        builtins.open = self._old_open

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
        with _TemporaryEvalEnvironment(output_dir), _Utf8DefaultOpen(), _WindowsSpiceNativeRuntime(output_dir):
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
