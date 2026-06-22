import json
from pathlib import Path


def test_implementation_manifest_maps_entrypoints_modules_symbols_and_tests():
    manifest = json.loads(Path("docs/runbooks/ACPR_FlowCal_V2_Implementation_Manifest.json").read_text(encoding="utf-8"))
    assert "python -m fate_x.engine.train_acpr_flowcal_v2" in manifest["formal_entrypoints"]
    assert "fate_x/acpr_flow_v2/model.py" in manifest["formal_modules"]
    assert "fate_x.acpr_flow_v2.model" in manifest["public_symbols"]
    assert manifest["tests_mapped_to_plan_sections"]
