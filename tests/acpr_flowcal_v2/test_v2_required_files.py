from pathlib import Path


def test_v2_required_files_exist():
    root = Path.cwd()
    required = [
        "fate_x/acpr_flow_v2/model.py",
        "fate_x/acpr_flow_v2/local_partial_transport.py",
        "fate_x/acpr_flow_v2/temporal_predicate_tracker.py",
        "fate_x/acpr_flow_v2/axis_aware_control_adapter.py",
        "fate_x/losses/acpr_flowcal_v2_losses.py",
        "fate_x/engine/train_acpr_flowcal_v2.py",
        "fate_x/engine/eval_acpr_flowcal_v2.py",
        "scripts/FATE_X_acpr_flowcal_v2_foreground.sh",
    ]
    missing = [p for p in required if not (root / p).exists()]
    assert not missing
