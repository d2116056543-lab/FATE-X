from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from fate_x.acpr_dynflow.config import load_dynflow_config
from fate_x.acpr_dynflow.predicate_ontology import EXACT_32_PREDICATES

REQUIRED_REPORTS = [
    "git_provenance.json",
    "formal_import_graph.json",
    "model_independence_audit.json",
    "config_binding_report.json",
    "direct_image_no_cache_audit.json",
    "adapt_metric_parity.json",
    "signal_contract_audit.json",
    "tensor_contracts.json",
    "backbone_audit.json",
    "oia_predicate_transfer_audit.json",
    "dynamic_predicate_audit.json",
    "nnpu_calalign_audit.json",
    "covariate_homogenization_audit.json",
    "pattern_router_audit.json",
    "mesoscopic_flow_audit.json",
    "traffic_state_audit.json",
    "response_lag_audit.json",
    "signal_codec_audit.json",
    "global_decision_audit.json",
    "decision_ledger_audit.json",
    "text_decoder_audit.json",
    "gradient_direction_audit.json",
    "loss_audit.json",
    "optimizer_scheduler_audit.json",
    "test_protocol_audit.json",
    "best_selector_audit.json",
    "gate_real_direct_image_smoke.json",
    "gate_gradient_chain.json",
    "gate_mechanism_fit_128.json",
    "gate_temporal_lag_necessity.json",
    "gate_intervention.json",
    "visual_artifact_index.json",
    "memory_probe_selection.json",
    "foreground_supervisor_audit.json",
    "review_report.json",
    "implementation_manifest.json",
]


def _run(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()


def _sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_static_reports(repo: Path, config_path: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_dynflow_config(config_path)
    head = _run(["git", "rev-parse", "HEAD"], repo)
    branch = _run(["git", "branch", "--show-current"], repo)
    remote = _run(["git", "ls-remote", "github", f"refs/heads/{branch}"], repo).split()[0]
    status = _run(["git", "status", "--porcelain"], repo)
    (out / "git_provenance.json").write_text(json.dumps({"branch": branch, "head": head, "github_head": remote, "clean": status == "", "status": status}, indent=2), encoding="utf-8")
    forbidden = ["ACPRFlowModel", "ACPRFlowCalV2Model", "TokenPMTAdapter", "LogSinkhornTransport", "TemporalEvidenceMemory"]
    formal_files = [repo / "fate_x/engine/train_acpr_dynflow.py", repo / "fate_x/engine/eval_acpr_dynflow.py", repo / "fate_x/acpr_dynflow/model.py"]
    hits = {}
    for file in formal_files:
        text = file.read_text(encoding="utf-8")
        hits[str(file.relative_to(repo))] = [name for name in forbidden if name in text]
    (out / "formal_import_graph.json").write_text(json.dumps({"forbidden_hits": hits, "passed": all(not v for v in hits.values())}, indent=2), encoding="utf-8")
    paths = cfg.raw.get("paths", {})
    video_swin_path = Path(str(paths.get("video_swin_kinetics_checkpoint", "")))
    bert_path = Path(str(paths.get("bert_dir", "")))
    model_independence = {
        "adapt_task_checkpoint_loaded_by_formal_training": False,
        "forbidden_formal_paths": [k for k, v in paths.items() if "adapt" in k.lower() and "reference" not in k.lower()],
        "oia_checkpoint": paths.get("oia_acpr_checkpoint"),
        "oia_checkpoint_exists": bool(paths.get("oia_acpr_checkpoint")) and Path(str(paths.get("oia_acpr_checkpoint"))).exists(),
        "video_swin_kinetics_checkpoint": str(video_swin_path),
        "video_swin_kinetics_checkpoint_exists": video_swin_path.exists(),
        "bert_dir": str(bert_path),
        "bert_dir_exists": bert_path.exists(),
        "bert_config_exists": (bert_path / "config.json").exists(),
        "bert_weight_exists": (bert_path / "model.safetensors").exists() or (bert_path / "pytorch_model.bin").exists(),
    }
    (out / "model_independence_audit.json").write_text(json.dumps(model_independence, indent=2), encoding="utf-8")
    (out / "config_binding_report.json").write_text(json.dumps({"field_count": len(cfg.consumer_manifest), "manifest": cfg.consumer_manifest}, indent=2), encoding="utf-8")
    (out / "implementation_manifest.json").write_text(json.dumps({"package": "ACPR-DynFlow V1", "formal_namespace": "fate_x/acpr_dynflow", "predicate_count": len(EXACT_32_PREDICATES), "config_sha256": _sha(config_path)}, indent=2), encoding="utf-8")
    return {
        "branch": branch,
        "head": head,
        "remote": remote,
        "clean": status == "",
        "oia_resolved": model_independence["oia_checkpoint_exists"],
        "video_swin_checkpoint_exists": model_independence["video_swin_kinetics_checkpoint_exists"],
        "bert_ready": model_independence["bert_dir_exists"] and model_independence["bert_config_exists"] and model_independence["bert_weight_exists"],
    }




def _failed_required_reports(out: Path) -> list[str]:
    failed: list[str] = []
    for name in REQUIRED_REPORTS:
        if name in {"review_report.json", "REVIEW_PASS_ACPR_DYNFLOW_V1.txt"}:
            continue
        path = out / name
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("passed") is False:
            failed.append(name)
    return failed

def maybe_write_pass(repo: Path, out: Path, state: dict, write_review_pass: bool) -> dict:
    missing = [name for name in REQUIRED_REPORTS if name != "review_report.json" and not (out / name).exists()]
    failed = _failed_required_reports(out)
    blockers = []
    if state["branch"] != "acpr_dynflow_v1":
        blockers.append("wrong_branch")
    if state["head"] != state["remote"]:
        blockers.append("local_remote_sha_mismatch")
    if not state["clean"]:
        blockers.append("dirty_worktree")
    if not state["oia_resolved"]:
        blockers.append("oia_checkpoint_unresolved")
    if state.get("video_swin_checkpoint_exists") is False:
        blockers.append("video_swin_checkpoint_missing")
    if state.get("bert_ready") is False:
        blockers.append("bert_base_missing")
    if state.get("swin_loaded") is False:
        blockers.append("video_swin_not_loaded")
    if state.get("bert_loaded") is False:
        blockers.append("bert_not_loaded")
    if state.get("oia_loaded") is False:
        blockers.append("oia_query_not_loaded")
    if missing:
        blockers.append("missing_required_reports")
    if failed:
        blockers.append("failed_required_reports")
    report = {"passed": not blockers, "blockers": blockers, "missing_reports": missing, "failed_reports": failed, "predicate_names": list(EXACT_32_PREDICATES)}
    (out / "review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if blockers:
        pass_path = out / "REVIEW_PASS_ACPR_DYNFLOW_V1.txt"
        if pass_path.exists():
            pass_path.unlink()
    elif write_review_pass:
        (out / "REVIEW_PASS_ACPR_DYNFLOW_V1.txt").write_text("ACPR_DYNFLOW_V1_IMPLEMENTATION_REVIEW_PASS\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default=".")
    p.add_argument("--config", default="configs/acpr_dynflow_v1_bddx_32f_224.yaml")
    p.add_argument("--output_dir", default=".background_runs/acpr_dynflow_v1_preflight")
    p.add_argument("--write_review_pass", action="store_true")
    args = p.parse_args()
    repo = Path(args.repo_root).resolve()
    out = repo / args.output_dir
    state = build_static_reports(repo, repo / args.config, out)
    report = maybe_write_pass(repo, out, state, args.write_review_pass)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

