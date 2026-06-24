import json

from fate_x.engine.eval_adapt_reference_dynflow import compare_metric_payloads


def test_metric_parity_requires_every_numeric_field_within_tolerance(tmp_path):
    original = {"CIDEr": 1.25, "Bleu_4": 0.3, "nested": {"RMSE": 2.5}}
    reproduced = {"CIDEr": 1.25 + 1e-8, "Bleu_4": 0.3, "nested": {"RMSE": 2.5}}
    report = compare_metric_payloads(original, reproduced, tolerance=1e-6)
    assert report["passed"] is True
    assert report["max_abs_error"] <= 1e-6

    bad = compare_metric_payloads(original, {**reproduced, "CIDEr": 1.3}, tolerance=1e-6)
    assert bad["passed"] is False
    assert "CIDEr" in bad["mismatches"]


def test_metric_parity_rejects_missing_fields():
    report = compare_metric_payloads({"CIDEr": 1.0, "SPICE": 0.2}, {"CIDEr": 1.0})
    assert report["passed"] is False
    assert report["missing_fields"] == ["SPICE"]
