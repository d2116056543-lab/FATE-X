from fate_x.acpr_flow_v2.config import FlowCalV2Config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.engine.train_acpr_flowcal_v2 import StageController


def test_stage_controller_maps_formal_epoch_boundaries():
    c = StageController(FlowCalV2Config())
    assert c.stage_for_epoch(0) == "semantic_recovery"
    assert c.stage_for_epoch(3) == "axis_aware_motion"
    assert c.stage_for_epoch(8) == "conflict_aware_joint"
    assert c.stage_for_epoch(13) == "explanation_scst"


def test_stage_controller_applies_real_freeze_lists():
    cfg = FlowCalV2Config(use_real_video_swin=False)
    model = ACPRFlowCalV2Model(cfg)
    controller = StageController(cfg)

    stage_r = controller.apply(model, 0)
    trainable_r = set(stage_r["trainable"])
    assert any(name.startswith("transport.") for name in trainable_r)
    assert any(name.startswith("memory.") for name in trainable_r)
    assert any(name.startswith("seca.query_explanation.") for name in trainable_r)
    assert any(name.startswith("seca.out_explanation.") for name in trainable_r)
    assert "seca.gate_explanation" in trainable_r
    assert not any(name.startswith("seca.query_action.") for name in trainable_r)
    assert not any(name.startswith("seca.out_action.") for name in trainable_r)
    assert "seca.gate_action" not in trainable_r
    assert not any(name.startswith("video.") for name in trainable_r)
    assert not any(name.startswith("motion.") for name in trainable_r)
    assert not any(name.startswith("control_adapter.") for name in trainable_r)

    stage_m = controller.apply(model, 3)
    trainable_m = set(stage_m["trainable"])
    assert any(name.startswith("control_adapter.") for name in trainable_m)
    assert any(name.startswith("flow.") for name in trainable_m)
    assert any(name.startswith("memory.") for name in trainable_m)
    assert not any(name.startswith("seca.") for name in trainable_m)
    assert not any(name.startswith("lm_head.") for name in trainable_m)

    stage_s = controller.apply(model, 13)
    trainable_s = set(stage_s["trainable"])
    assert any(name.startswith("lm_head.") for name in trainable_s)
    assert any(name.startswith("seca.") for name in trainable_s)
    assert not any(name.startswith("lane_flow.") for name in trainable_s)
    assert not any(name.startswith("control_adapter.") for name in trainable_s)
