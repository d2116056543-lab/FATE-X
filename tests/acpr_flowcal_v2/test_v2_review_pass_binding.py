from fate_x.engine.audit_acpr_flowcal_v2 import run_static_contract_audit


def test_review_pass_requires_manifest_to_be_present():
    report = run_static_contract_audit(".")
    assert report["manifest"]["present"] is True
    assert not report["manifest"]["missing_symbols"]

def test_review_pass_requires_dynamic_preflight_gates(tmp_path):
    from fate_x.engine.audit_acpr_flowcal_v2 import can_write_review_pass

    static_report = {
        'forbidden_imports': [],
        'missing_required_files': [],
        'manifest': {'present': True, 'missing_files': [], 'missing_symbols': [], 'errors': []},
    }
    assert not can_write_review_pass(static_report, tmp_path)
    (tmp_path / 'preflight_gates.json').write_text('{"all_gates_passed": false, "review_pass_authorized": false}', encoding='utf-8')
    assert not can_write_review_pass(static_report, tmp_path)
    (tmp_path / 'preflight_gates.json').write_text('{"all_gates_passed": true, "review_pass_authorized": true}', encoding='utf-8')
    assert can_write_review_pass(static_report, tmp_path)
