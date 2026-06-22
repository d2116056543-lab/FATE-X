import json

import torch

from fate_x.engine.run_acpr_flowcal_v2_preflight import _json_safe


def test_preflight_json_safe_converts_nested_tensors():
    payload = {"x": torch.tensor([1.0, 2.0]), "nested": {"y": torch.tensor(3.0)}}
    converted = _json_safe(payload)
    json.dumps(converted)
    assert converted["x"] == [1.0, 2.0]
    assert converted["nested"]["y"] == 3.0

def test_dynamic_preflight_reports_all_named_gates(tmp_path):
    from fate_x.engine.run_acpr_flowcal_v2_preflight import run_dynamic_preflight

    report = run_dynamic_preflight(
        repo='.',
        output_dir=tmp_path,
        device='cpu',
        synthetic=True,
        require_all=False,
    )
    gate_names = {gate['name'] for gate in report['gates']}
    assert gate_names == {
        'A_compile_imports_tests',
        'B_adapt_equivalence',
        'C_direct_image_smoke',
        'D_gradient_chain',
        'E_stage_execution',
        'F_mechanism_fit',
        'G_temporal_necessity',
        'H_real_intervention',
        'I_memory_selection',
        'J_visualization',
    }
    assert (tmp_path / 'preflight_gates.json').exists()
    assert report['review_pass_authorized'] is False
