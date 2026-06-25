from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORBIDDEN = (
    "fate_x.acpr_dynflow",
    "fate_x.acpr_flow",
    "fate_x.acpr_flow_v2",
    "fate_x.models.flowtrace_pmt_model",
    "fate_x.models.token_pmt_adapter",
    "fate_x.models.sinkhorn_transport",
)

REQUIRED_REPORTS = (
    "git_provenance.json",
    "formal_import_graph.json",
    "config_binding_report.json",
    "model_independence_audit.json",
    "direct_image_no_cache_audit.json",
    "adapt_metric_parity.json",
    "signal_contract_audit.json",
    "tensor_contracts.json",
    "video_swin_backbone_audit.json",
    "oia_transfer_audit.json",
    "dynamic_predicate_audit.json",
    "nnpu_calalign_audit.json",
    "semantic_consolidation_audit.json",
    "corridor_flow_audit.json",
    "pattern_traffic_audit.json",
    "response_lag_audit.json",
    "query_motion_transformer_audit.json",
    "decision_ledger_audit.json",
    "text_decoder_audit.json",
    "gradient_direction_audit.json",
    "loss_audit.json",
    "optimizer_precision_scheduler_audit.json",
    "throughput_memory_probe.json",
    "test_protocol_audit.json",
    "best_selector_audit.json",
    "gate_real_direct_image_smoke.json",
    "gate_gradient_chain.json",
    "gate_mechanism_fit_128.json",
    "gate_identity_checks.json",
    "gate_temporal_lag.json",
    "intervention_audit.json",
    "visual_artifact_index.json",
    "foreground_supervisor_audit.json",
    "implementation_manifest.json",
    "review_report.json",
)


def audit_import_graph(package: str = "fate_x/acpr_dynflow_swin") -> dict[str, Any]:
    offenders = []
    for file in Path(package).rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name == bad or alias.name.startswith(bad + ".") for bad in FORBIDDEN):
                        offenders.append({"file": str(file), "module": alias.name})
            if module and any(module == bad or module.startswith(bad + ".") for bad in FORBIDDEN):
                offenders.append({"file": str(file), "module": module})
    return {"passed": not offenders, "offenders": offenders}


def _blocker(code: str, message: str, path: str | None = None) -> dict[str, str]:
    item = {"code": code, "message": message}
    if path:
        item["path"] = path
    return item


def _read(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _contains_any(path: str, patterns: tuple[str, ...]) -> bool:
    text = _read(path).lower()
    return any(pattern.lower() in text for pattern in patterns)


def _windows_gitdir_to_wsl(path_text: str) -> str:
    text = path_text.strip().replace("\\", "/")
    if len(text) > 2 and text[1:3] == ":/":
        return f"/mnt/{text[0].lower()}/{text[3:]}"
    return text


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        git_file = Path(".git")
        if not git_file.is_file():
            raise
        text = git_file.read_text(encoding="utf-8").strip()
        if not text.lower().startswith("gitdir:"):
            raise
        env = os.environ.copy()
        env["GIT_DIR"] = _windows_gitdir_to_wsl(text.split(":", 1)[1])
        env["GIT_WORK_TREE"] = str(Path.cwd())
        return subprocess.check_output(["git", *args], text=True, env=env, stderr=subprocess.DEVNULL).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_review_pass(path: str | Path, expected_head: str) -> dict[str, Any]:
    blockers: list[str] = []
    review_path = Path(path)
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        return {"passed": False, "blockers": ["invalid_review_pass"], "payload": {}}
    if payload.get("authorization") != "ACPR_DYNFLOW_SWIN_V1_IMPLEMENTATION_REVIEW_PASS":
        blockers.append("authorization_missing")
    if not str(payload.get("reviewer", "")).strip():
        blockers.append("reviewer_missing")
    if payload.get("local_head") != expected_head or payload.get("github_head") != expected_head:
        blockers.append("sha_mismatch")
    if payload.get("clean") is not True:
        blockers.append("worktree_not_clean")
    if payload.get("all_reports_passed") is not True:
        blockers.append("reports_not_passed")
    return {"passed": not blockers, "blockers": blockers, "payload": payload}


def _report_passed(payload: dict[str, Any]) -> bool:
    passed = payload.get("passed", payload.get("pass")) is True
    status = payload.get("status", payload.get("review_status"))
    return passed or status in {"pass", "passed"}


def reports_ready_for_review_pass(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    missing: list[str] = []
    invalid: list[str] = []
    not_passed: list[str] = []
    ignored_self_gate: list[str] = []
    report_sha256: dict[str, str] = {}
    for name in REQUIRED_REPORTS:
        path = root / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            invalid.append(name)
            continue
        if _report_passed(payload):
            report_sha256[name] = _sha256(path)
            continue
        if name == "review_report.json":
            ignored_self_gate.append(name)
            report_sha256[name] = _sha256(path)
            continue
        not_passed.append(name)
    return {
        "passed": not missing and not invalid and not not_passed,
        "missing": missing,
        "invalid": invalid,
        "not_passed": not_passed,
        "ignored_self_gate": ignored_self_gate,
        "report_sha256": report_sha256,
    }


def write_review_pass(output_dir: str | Path, reviewer: str) -> Path:
    if not reviewer.strip():
        raise ValueError("independent reviewer identity is required")
    root = Path(output_dir)
    readiness = reports_ready_for_review_pass(root)
    if not readiness["passed"]:
        failed = readiness["missing"] + readiness["invalid"] + readiness["not_passed"]
        raise RuntimeError(f"cannot authorize review pass: reports are not ready: {', '.join(failed)}")
    local_head = _git(["rev-parse", "HEAD"])
    github_head = _git(["ls-remote", "github", "refs/heads/acpr_dynflow_v1"]).split()[0]
    status = _git(["status", "--porcelain"])
    if status or local_head != github_head:
        raise RuntimeError("cannot authorize review pass: clean local/GitHub SHA gate failed")
    payload = {
        "authorization": "ACPR_DYNFLOW_SWIN_V1_IMPLEMENTATION_REVIEW_PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "worktree": str(Path.cwd().resolve()),
        "branch": _git(["branch", "--show-current"]),
        "local_head": local_head,
        "github_head": github_head,
        "clean": True,
        "all_reports_passed": True,
        "ignored_self_gate": readiness["ignored_self_gate"],
        "report_sha256": readiness["report_sha256"],
    }
    path = root / "REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_blocking_audit(
    config: str = "configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml",
    output_dir: str | None = ".background_runs/acpr_dynflow_swin_v1_preflight",
    package: str = "fate_x/acpr_dynflow_swin",
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    import_report = audit_import_graph(package)
    if not import_report["passed"]:
        blockers.append(_blocker("formal_import_graph_forbidden_import", "formal namespace imports a forbidden legacy module"))

    if not Path("fate_x/acpr_dynflow_swin/signal_codec.py").exists():
        blockers.append(_blocker("signal_codec_missing", "BDD-X signal codec file is missing"))

    evaluator = _read("fate_x/engine/eval_acpr_dynflow_swin.py")
    if "full evaluator requires dataset assets" in evaluator or "implemented as a library entrypoint" in evaluator:
        blockers.append(_blocker("adapt_metric_parity_missing", "formal evaluator does not run exact ADAPT text/control metrics"))
    if "run_adapt_sep_caption_eval" not in evaluator:
        blockers.append(_blocker("adapt_metric_bridge_not_called", "ADAPT caption metric bridge is not called"))

    preflight = _read("fate_x/engine/run_acpr_dynflow_swin_preflight.py")
    if "smoke_batch" in preflight or "review pass requires full dynamic gates" in preflight:
        blockers.append(_blocker("preflight_smoke_not_full_gates", "preflight still relies on a smoke batch instead of all blocking gates"))

    trainer = _read("fate_x/engine/train_acpr_dynflow_swin.py")
    if "evaluate(" not in trainer or "full test" not in trainer.lower():
        blockers.append(_blocker("trainer_missing_per_epoch_full_test_eval", "trainer does not run full test text/control evaluation after every epoch"))
    if "torch.autocast" not in trainer or "bfloat16" not in trainer:
        blockers.append(_blocker("trainer_missing_bf16_runtime", "trainer does not prove BF16 autocast runtime"))
    if "checkpoint_best_test.pth" not in trainer:
        blockers.append(_blocker("trainer_missing_best_test_checkpoint", "trainer does not maintain checkpoint_best_test.pth"))

    if _contains_any("fate_x/engine/export_acpr_dynflow_swin_visuals.py", ("requires a reviewed checkpoint",)):
        blockers.append(_blocker("visual_export_scaffold_or_placeholder", "visual export script is still a scaffold"))
    if _contains_any("fate_x/engine/build_acpr_dynflow_swin_atlas.py", ("requires reviewed visual artifacts",)):
        blockers.append(_blocker("atlas_scaffold_or_placeholder", "atlas build script is still a scaffold"))
    if _contains_any("fate_x/explain/acpr_dynflow_swin_renderer.py", ("1x1 png", "fallback", "json.dumps(case")):
        blockers.append(_blocker("canvas_renderer_placeholder", "renderer is still template/dump based"))

    if _contains_any("fate_x/engine/eval_adapt_reference_dynflow.py", ("comparison-only and never loaded",)):
        blockers.append(_blocker("adapt_reference_eval_scaffold", "ADAPT reference evaluator is a scaffold"))

    if output_dir:
        root = Path(output_dir)
        missing = [name for name in REQUIRED_REPORTS if not (root / name).exists()]
        if missing:
            blockers.append(_blocker("preflight_required_reports_missing", f"missing required preflight reports: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}"))
        else:
            not_passed = []
            for name in REQUIRED_REPORTS:
                try:
                    payload = json.loads((root / name).read_text(encoding="utf-8"))
                except Exception:
                    not_passed.append(name)
                    continue
                if not _report_passed(payload):
                    not_passed.append(name)
            if not_passed and not not_passed == ["review_report.json"]:
                blockers.append(_blocker("preflight_dynamic_gates_not_passed", f"preflight reports are present but not passed: {', '.join(not_passed[:8])}{'...' if len(not_passed) > 8 else ''}"))
    else:
        blockers.append(_blocker("preflight_dynamic_gates_not_passed", "no preflight output_dir was provided; dynamic gate reports cannot be verified"))

    report = {
        "passed": not blockers,
        "config": config,
        "import_graph": import_report,
        "blockers": blockers,
        "review_pass_authorized": False,
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "formal_import_graph.json").write_text(json.dumps(import_report, indent=2), encoding="utf-8")
        (out / "review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        stale = out / "REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt"
        if stale.exists() and blockers:
            stale.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--package", default="fate_x/acpr_dynflow_swin")
    parser.add_argument("--output_dir", default=".background_runs/acpr_dynflow_swin_v1_preflight")
    parser.add_argument("--write_review_pass", action="store_true")
    parser.add_argument("--reviewer", default="")
    args = parser.parse_args()
    result = run_blocking_audit(args.config, args.output_dir, args.package)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
    if args.write_review_pass:
        print(write_review_pass(args.output_dir, args.reviewer))


if __name__ == "__main__":
    main()
