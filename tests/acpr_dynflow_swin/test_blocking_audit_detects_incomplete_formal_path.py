from __future__ import annotations


def test_blocking_audit_reports_required_dynamic_gate_blockers():
    from fate_x.engine.audit_acpr_dynflow_swin import run_blocking_audit

    report = run_blocking_audit(
        config="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml",
        output_dir=None,
    )
    codes = {item["code"] for item in report["blockers"]}
    assert report["passed"] is False
    assert "preflight_dynamic_gates_not_passed" in codes
    assert report["review_pass_authorized"] is False
