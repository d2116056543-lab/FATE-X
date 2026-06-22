from pathlib import Path

from fate_x.engine.audit_acpr_flowcal_v2 import run_static_contract_audit


def test_static_contract_rejects_legacy_imports(tmp_path):
    root = tmp_path / "repo"
    (root / "fate_x/acpr_flow_v2").mkdir(parents=True)
    (root / "fate_x/acpr_flow_v2/model.py").write_text(
        "from fate_x.acpr_flow.model import ACPRFlowModel\n", encoding="utf-8"
    )
    report = run_static_contract_audit(root)
    assert report["forbidden_imports"], report


def test_static_contract_accepts_v2_namespace_without_legacy_imports(tmp_path):
    root = tmp_path / "repo"
    (root / "fate_x/acpr_flow_v2").mkdir(parents=True)
    (root / "fate_x/acpr_flow_v2/model.py").write_text("import torch\n", encoding="utf-8")
    report = run_static_contract_audit(root)
    assert not report["forbidden_imports"], report
