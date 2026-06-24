from __future__ import annotations

import json
from pathlib import Path

from fate_x.engine.run_acpr_dynflow_swin_preflight import merge_external_reports


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_external_reports_copies_real_reports_and_marks_missing_without_placeholder(tmp_path: Path):
    out = tmp_path / "preflight"
    runtime = tmp_path / "runtime"
    parity = tmp_path / "parity"
    throughput = tmp_path / "throughput.json"
    mechanism = tmp_path / "mechanism.json"
    _write(runtime / "tensor_contracts.json", {"passed": True, "status": "pass", "evidence": "tensor"})
    _write(parity / "adapt_metric_parity.json", {"passed": True, "status": "pass", "max_abs_error": 0.0})
    _write(throughput, {"passed": True, "status": "pass", "selected_candidate": {"batch_size": 4}})
    _write(mechanism, {"passed": True, "status": "pass", "sample_count": 128})
    reports = {}
    merge_external_reports(
        reports,
        out,
        runtime_audit_dir=str(runtime),
        adapt_parity_dir=str(parity),
        throughput_report=str(throughput),
        mechanism_report=str(mechanism),
    )
    assert reports["tensor_contracts.json"]["evidence"] == "tensor"
    assert reports["adapt_metric_parity.json"]["max_abs_error"] == 0.0
    assert reports["throughput_memory_probe.json"]["selected_candidate"]["batch_size"] == 4
    assert reports["gate_mechanism_fit_128.json"]["sample_count"] == 128
    assert "requires real" not in json.dumps(reports).lower()
