import json
from pathlib import Path

from fate_x.engine.audit_acpr_dynflow_swin import REQUIRED_REPORTS, reports_ready_for_review_pass


def test_reports_ready_allows_only_review_report_blocked(tmp_path: Path) -> None:
    for name in REQUIRED_REPORTS:
        payload = {"status": "pass", "passed": True}
        if name == "review_report.json":
            payload = {
                "status": "blocked",
                "passed": False,
                "blockers": [{"code": "preflight_dynamic_gates_not_passed"}],
            }
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    report = reports_ready_for_review_pass(tmp_path)

    assert report["passed"] is True
    assert report["ignored_self_gate"] == ["review_report.json"]
    assert report["not_passed"] == []


def test_reports_ready_rejects_non_review_blockers(tmp_path: Path) -> None:
    for name in REQUIRED_REPORTS:
        payload = {"status": "pass", "passed": True}
        if name == "gate_mechanism_fit_128.json":
            payload = {"status": "blocked", "passed": False}
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    report = reports_ready_for_review_pass(tmp_path)

    assert report["passed"] is False
    assert report["not_passed"] == ["gate_mechanism_fit_128.json"]
