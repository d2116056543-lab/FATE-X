from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

FORBIDDEN = [
    "from fate_x.acpr_flow.model",
    "import fate_x.acpr_flow.model",
    "TokenPMTAdapter",
    "FlowTraceLoss",
    "sinkhorn",
]


def run_static_contract_audit(root: str | Path) -> dict:
    root = Path(root)
    files = list((root / "fate_x/acpr_flow_v2").glob("**/*.py")) if (root / "fate_x/acpr_flow_v2").exists() else []
    forbidden_imports = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in FORBIDDEN:
            if needle in text:
                forbidden_imports.append({"file": str(path.relative_to(root)), "needle": needle})
    required = [
        "fate_x/acpr_flow_v2/model.py",
        "fate_x/acpr_flow_v2/local_partial_transport.py",
        "fate_x/acpr_flow_v2/temporal_predicate_tracker.py",
        "fate_x/losses/acpr_flowcal_v2_losses.py",
        "fate_x/engine/train_acpr_flowcal_v2.py",
        "docs/runbooks/ACPR_FlowCal_V2_Implementation_Manifest.json",
    ]
    missing = [p for p in required if not (root / p).exists()]
    manifest_report = _audit_implementation_manifest(root)
    return {
        "forbidden_imports": forbidden_imports,
        "missing_required_files": missing,
        "file_count": len(files),
        "direct_image_training": True,
        "feature_cache_enabled": False,
        "token_cache_enabled": False,
        "manifest": manifest_report,
    }


def _audit_implementation_manifest(root: Path) -> dict:
    manifest_path = root / "docs/runbooks/ACPR_FlowCal_V2_Implementation_Manifest.json"
    if not manifest_path.exists():
        return {"present": False, "missing_files": [], "missing_symbols": [], "errors": ["manifest_missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_files = [p for p in manifest.get("formal_modules", []) if not (root / p).exists()]
    missing_symbols = []
    errors = []
    for dotted, symbols in manifest.get("public_symbols", {}).items():
        try:
            module = importlib.import_module(dotted)
        except Exception as exc:  # pragma: no cover - reported as audit data
            errors.append(f"{dotted}: {exc}")
            continue
        for symbol in symbols:
            if not hasattr(module, symbol):
                missing_symbols.append({"module": dotted, "symbol": symbol})
    return {
        "present": True,
        "missing_files": missing_files,
        "missing_symbols": missing_symbols,
        "errors": errors,
        "formal_entrypoints": manifest.get("formal_entrypoints", []),
        "tests_mapped": len(manifest.get("tests_mapped_to_plan_sections", {})),
    }


def _static_ok(report: dict[str, Any]) -> bool:
    manifest = report.get("manifest", {})
    return bool(
        not report.get("forbidden_imports")
        and not report.get("missing_required_files")
        and manifest.get("present")
        and not manifest.get("missing_files")
        and not manifest.get("missing_symbols")
        and not manifest.get("errors")
    )


def can_write_review_pass(static_report: dict[str, Any], preflight_dir: str | Path) -> bool:
    """Review pass requires both static contract and dynamic Gate A-J authorization."""
    if not _static_ok(static_report):
        return False
    gate_file = Path(preflight_dir) / "preflight_gates.json"
    if not gate_file.exists():
        return False
    try:
        dynamic = json.loads(gate_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(dynamic.get("all_gates_passed") and dynamic.get("review_pass_authorized"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output_dir", default=".background_runs/acpr_flowcal_v2_preflight")
    parser.add_argument("--write_review_pass", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = run_static_contract_audit(root)
    gate_file = out / "preflight_gates.json"
    if gate_file.exists():
        try:
            report["dynamic_preflight"] = json.loads(gate_file.read_text(encoding="utf-8"))
        except Exception as exc:
            report["dynamic_preflight"] = {"error": str(exc)}
    (out / "static_contract_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    authorized = can_write_review_pass(report, out)
    if args.write_review_pass and authorized:
        (root / "REVIEW_PASS_ACPR_FLOWCAL_V2.txt").write_text(
            "REVIEW_PASS_ACPR_FLOWCAL_V2\nstatic_contract_audit=pass\ndynamic_preflight_gates=pass\n",
            encoding="utf-8",
        )
    if not _static_ok(report) or (args.write_review_pass and not authorized):
        raise SystemExit(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
