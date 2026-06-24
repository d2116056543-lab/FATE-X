
from pathlib import Path
import json


FORBIDDEN = ("Start-Process", "Start-Job", "schtasks", "nohup", "DETACHED_PROCESS", "WindowStyle Hidden")


def test_foreground_scripts_do_not_detach():
    for path in [
        Path("scripts/FATE_X_acpr_dynflow_swin_v1_foreground.ps1"),
        Path("scripts/FATE_X_acpr_dynflow_swin_v1_foreground.sh"),
        Path("fate_x/engine/supervise_acpr_dynflow_swin_foreground.py"),
    ]:
        assert path.exists(), f"{path} missing"
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in FORBIDDEN), f"{path} contains detached execution token"


def test_review_pass_validation_rejects_stale_sha(tmp_path):
    from fate_x.engine.audit_acpr_dynflow_swin import verify_review_pass

    path = tmp_path / "REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt"
    path.write_text(
        json.dumps(
            {
                "authorization": "ACPR_DYNFLOW_SWIN_V1_IMPLEMENTATION_REVIEW_PASS",
                "reviewer": "agent-b",
                "local_head": "old",
                "github_head": "old",
                "clean": True,
                "all_reports_passed": True,
            }
        ),
        encoding="utf-8",
    )
    result = verify_review_pass(path, expected_head="new")
    assert result["passed"] is False
    assert "sha_mismatch" in result["blockers"]


def test_review_pass_validation_requires_independent_reviewer(tmp_path):
    from fate_x.engine.audit_acpr_dynflow_swin import verify_review_pass

    path = tmp_path / "REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt"
    path.write_text(
        json.dumps(
            {
                "authorization": "ACPR_DYNFLOW_SWIN_V1_IMPLEMENTATION_REVIEW_PASS",
                "reviewer": "",
                "local_head": "abc",
                "github_head": "abc",
                "clean": True,
                "all_reports_passed": True,
            }
        ),
        encoding="utf-8",
    )
    result = verify_review_pass(path, expected_head="abc")
    assert result["passed"] is False
    assert "reviewer_missing" in result["blockers"]
