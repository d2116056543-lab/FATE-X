from fate_x.acpr_flow.free_text_partial_targets import FreeTextPartialTargetBuilder


def test_free_text_targets_use_unknown_low_weight_not_hard_negative():
    builder = FreeTextPartialTargetBuilder(unknown_negative_weight=0.075)
    out = builder.build(["maintain"], ["traffic is clear"])
    assert out["predicate_positive"][0, builder.pred_index["road_clear"]] == 1
    assert out["predicate_contradiction"][0, builder.pred_index["road_crowded"]] == 1
    unknown_idx = builder.pred_index["cyclist_front"]
    assert out["predicate_known_mask"][0, unknown_idx] == 0
    assert abs(float(out["predicate_reliability"][0, unknown_idx]) - 0.075) < 1e-6


def test_free_text_targets_recognize_stopped_cars():
    builder = FreeTextPartialTargetBuilder()
    out = builder.build(["slow"], ["cars ahead are stopped"])
    assert out["flow_positive"][0, builder.flow_index["queue_congestion"]] == 1
    assert out["flow_contradiction"][0, builder.flow_index["clear_open_flow"]] == 1
