from pathlib import Path

from fate_x.engine.supervise_acpr_flowcal_foreground import scan_no_detach


def test_foreground_scripts_do_not_use_detached_processes():
    for path in [
        Path("scripts/FATE_X_acpr_flowcal_pp_v1_foreground.ps1"),
        Path("scripts/FATE_X_acpr_flowcal_pp_v1_foreground.sh"),
    ]:
        assert scan_no_detach(path) == []
        text = path.read_text(encoding="utf-8")
        assert "audit_acpr_flowcal_pp" in text
        assert "train_acpr_flowcal_pp" in text
        if path.suffix == ".ps1":
            assert "RequireReviewPass" in text
            assert "REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt" in text
