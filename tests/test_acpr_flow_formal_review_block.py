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
    assert "formal pretrained ADAPT Video Swin backbone is not loaded" in blockers
    assert "formal review pass requires cuda direct-image execution" in blockers
    assert "missing formal preflight evidence: gate_b_direct_image_8step_smoke.json" in blockers
    assert "missing formal preflight evidence: gate_d_mechanism_overfit_128_report.json" in blockers
    assert "Temporal HardPair is not integrated into the formal model/trainer path" not in blockers
    assert "Sequence-CalAlign is not integrated into the formal trainer/evaluator path" not in blockers
    assert "formal model/trainer does not route text generation through BertForImageCaptioning ACPR SECA hook" not in blockers
    assert "missing formal preflight evidence: foreground_supervisor_smoke.json" in blockers
    assert "missing_gate_b_8_step_direct_image_smoke" in blockers
    assert "missing_gate_c_gradient_chain" in blockers
    assert "missing_gate_d_128_sample_mechanism_overfit" in blockers
    assert "missing_hardpair_integration" not in blockers
    assert "missing_sequence_calalign_integration" not in blockers
    assert "formal_path_not_using_bert_captioning_seca" not in blockers
