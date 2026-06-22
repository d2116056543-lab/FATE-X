from pathlib import Path


def test_supervisor_mentions_review_pass_requirement():
    text = Path("fate_x/engine/supervise_acpr_flowcal_v2_foreground.py").read_text(encoding="utf-8")
    assert "require_review_pass" in text
