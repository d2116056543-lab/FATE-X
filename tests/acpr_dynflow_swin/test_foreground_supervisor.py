
from pathlib import Path


FORBIDDEN = ("Start-Process", "Start-Job", "schtasks", "nohup", "DETACHED_PROCESS", "WindowStyle Hidden")


def test_foreground_scripts_do_not_detach():
    for path in [
        Path("scripts/FATE_X_acpr_dynflow_swin_v1_foreground.ps1"),
        Path("scripts/FATE_X_acpr_dynflow_swin_v1_foreground.sh"),
        Path("fate_x/engine/supervise_acpr_dynflow_swin_foreground.py"),
    ]:
        assert path.exists(), f"{path} missing"
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in FORBIDDEN), f"{path} contains detached execution token"
