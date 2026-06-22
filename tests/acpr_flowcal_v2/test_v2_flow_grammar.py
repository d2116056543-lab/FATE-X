from fate_x.acpr_flow_v2.axis_aware_flow_composer import AxisAwareFlowComposer


def test_flow_composer_preserves_13_semantic_factor_names():
    model = AxisAwareFlowComposer(dim=4)
    assert len(model.semantic_names) == 13
    assert "queue_congestion" in model.semantic_names
