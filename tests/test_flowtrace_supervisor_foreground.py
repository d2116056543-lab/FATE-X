from pathlib import Path


def test_supervisor_is_not_detached():
    text = Path("fate_x/engine/supervise_flowtrace_foreground.py").read_text() + Path("scripts/FATE_X_flowtrace_pmt_v1_foreground.ps1").read_text()
    forbidden = ["Start-Process", "Start-Job", "schtasks", "nohup", "DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"]
    for token in forbidden:
        assert token not in text
