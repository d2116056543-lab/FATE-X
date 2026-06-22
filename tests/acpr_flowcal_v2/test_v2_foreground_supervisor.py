from pathlib import Path


def test_foreground_supervisor_does_not_use_detached_process_primitives():
    text = Path("fate_x/engine/supervise_acpr_flowcal_v2_foreground.py").read_text(encoding="utf-8")
    forbidden = ["Start-Process", "Start-Job", "schtasks", "DETACHED_PROCESS", "daemon=True"]
    assert not [x for x in forbidden if x in text]
