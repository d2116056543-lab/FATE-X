from fate_x.acpr_flow_v2.config import ACPRFlowCalV2Config
from fate_x.engine.probe_acpr_flowcal_v2_memory import run_probe


def test_memory_probe_does_not_mark_unmeasured_candidates_stable(tmp_path):
    cfg = ACPRFlowCalV2Config(
        hidden_dim=16,
        state_dim=16,
        text_hidden_dim=32,
        text_vocab_size=101,
        num_frames=4,
        use_real_video_swin=False,
    )

    result = run_probe(
        str(tmp_path),
        candidates=[{"batch_size": 2, "gradient_accumulation_steps": 16}],
        device="cpu",
        synthetic=True,
        synthetic_config=cfg,
        warmup_steps=0,
        measured_steps=1,
    )

    candidate = result["candidates"][0]
    assert candidate["measured_steps"] == 1
    assert "finite" in candidate
    assert "peak_reserved_gib" in candidate
    assert result["selected"] == candidate["candidate"]
    assert result["review_pass_eligible"] is False
