from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from fate_x.acpr_dynflow_swin.config import load_config, build_config_consumer_manifest
from fate_x.engine.audit_acpr_dynflow_swin import REQUIRED_REPORTS, audit_import_graph


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _blocked(name: str, reason: str) -> dict[str, Any]:
    return {"report": name, "status": "blocked", "passed": False, "reason": reason}


def run_preflight(config: str, output_dir: str) -> dict[str, Any]:
    cfg = load_config(config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import_graph = audit_import_graph()
    try:
        branch = _git(["branch", "--show-current"])
        head = _git(["rev-parse", "HEAD"])
        remote = _git(["ls-remote", "github", "refs/heads/acpr_dynflow_v1"]).split()[0]
        status = _git(["status", "--porcelain"])
    except Exception as exc:
        branch = head = remote = ""
        status = f"git_error={exc}"

    reports: dict[str, dict[str, Any]] = {
        "git_provenance.json": {
            "status": "pass" if branch == "acpr_dynflow_v1" and head and head == remote and not status else "blocked",
            "passed": bool(branch == "acpr_dynflow_v1" and head and head == remote and not status),
            "branch": branch,
            "head": head,
            "github_head": remote,
            "status_short": status,
        },
        "formal_import_graph.json": {"status": "pass" if import_graph["passed"] else "blocked", "passed": import_graph["passed"], **import_graph},
        "config_binding_report.json": {"status": "pass", "passed": True, "manifest": build_config_consumer_manifest(cfg)},
        "implementation_manifest.json": {
            "status": "blocked",
            "passed": False,
            "config_sha256": _sha256(Path(config)),
            "reason": "full file-level implementation evidence is not complete",
        },
    }
    dynamic_block_reason = "requires real WSL/Linux CUDA direct-image gate execution; this preflight records blocker rather than fabricating evidence"
    for name in REQUIRED_REPORTS:
        reports.setdefault(name, _blocked(name, dynamic_block_reason))
    reports["review_report.json"] = {
        "status": "blocked",
        "passed": False,
        "review_pass_authorized": False,
        "reason": "not all dynamic ACPR-DynFlow-Swin gates have passed",
        "required_reports": list(REQUIRED_REPORTS),
    }
    for name, payload in reports.items():
        _json(out_dir / name, payload)
    summary = {
        "status": "blocked",
        "passed": False,
        "output_dir": str(out_dir),
        "blocked_reports": [name for name, payload in reports.items() if payload.get("passed") is not True],
    }
    _json(out_dir / "preflight_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--output_dir", default=".background_runs/acpr_dynflow_swin_v1_preflight")
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
