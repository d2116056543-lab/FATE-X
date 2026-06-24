from fate_x.engine.fit_acpr_dynflow_swin_mechanism import summarize_mechanism_fit


def test_mechanism_fit_summary_requires_real_improvement_and_noncollapse():
    initial = {
        "total_loss": 10.0,
        "final_speed_normalized": 4.0,
        "final_course_normalized": 3.0,
        "action_text": 2.0,
        "explanation_text": 1.0,
        "predicate_nnpu": 0.5,
        "pattern_semantic": 0.4,
        "traffic_state_semantic": 0.3,
        "contribution_alignment": 0.2,
    }
    final = {
        "total_loss": 8.0,
        "final_speed_normalized": 3.0,
        "final_course_normalized": 2.0,
        "action_text": 1.5,
        "explanation_text": 0.8,
        "predicate_nnpu": 0.4,
        "pattern_semantic": 0.3,
        "traffic_state_semantic": 0.2,
        "contribution_alignment": 0.1,
    }
    collapse = {
        "predicate_std": 0.02,
        "pattern_std": 0.03,
        "factor_std": 0.04,
        "benefit_gate_mean_abs": 0.2,
        "course_lateral_effect_mean_abs": 0.1,
        "flow_contribution_mean_abs": 0.1,
    }
    report = summarize_mechanism_fit(
        initial,
        final,
        collapse,
        sample_count=128,
        optimizer_steps=8,
    )
    assert report["passed"] is True
    assert report["improvements"]["total_loss"] == 2.0

    failed = summarize_mechanism_fit(
        initial,
        {**final, "predicate_nnpu": 0.8},
        {**collapse, "pattern_std": 0.0},
        sample_count=128,
        optimizer_steps=8,
    )
    assert failed["passed"] is False
    assert "predicate_nnpu" in failed["failed_improvements"]
    assert "pattern_not_constant" in failed["failed_collapse_checks"]
