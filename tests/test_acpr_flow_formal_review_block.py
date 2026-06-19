import pytest

from fate_x.engine import audit_acpr_flowcal_pp


def test_write_review_pass_is_blocked_while_formal_gates_have_blockers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audit_acpr_flowcal_pp,
        "git_guard",
        lambda *args, **kwargs: {
            "branch": "flowtrace_pmt_v1",
            "dirty_status": "",
            "git_head": "local",
            "github_remote_head": "local",
        },
    )

    with pytest.raises(RuntimeError, match="formal review pass blocked"):
        audit_acpr_flowcal_pp.run_audit(
            "configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml",
            str(tmp_path),
            device="cpu",
            write_review_pass=True,
        )

    assert not (tmp_path / "REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt").exists()
    blockers = (tmp_path / "formal_gate_blockers.json").read_text(encoding="utf-8")
    assert "TinyDirectImageVideoBackbone" in blockers
    assert "random frame tensors" in blockers
