from fate_x.acpr_flow_v2.config import FlowCalV2Config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.engine.train_acpr_flowcal_v2 import build_optimizer_groups


def test_optimizer_groups_separate_decay_and_no_decay():
    groups = build_optimizer_groups(ACPRFlowCalV2Model(FlowCalV2Config(hidden_dim=8, text_vocab_size=11)), FlowCalV2Config())
    assert groups
    assert {g["weight_decay"] for g in groups} == {0.0, 0.01}
    assert any(g.get("lr_key") == "action_seca" for g in groups)
    assert any(g.get("lr_key") == "explanation_seca" for g in groups)
