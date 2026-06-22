from fate_x.acpr_flow_v2.contextual_reason_target import FrozenContextualReasonTarget


def test_contextual_reason_target_encodes_action_and_justification_without_grad():
    target = FrozenContextualReasonTarget(dim=8)
    out = target.build_target(["slow"], ["traffic ahead"])
    assert out["target"].shape[-1] == 8
    assert out["target"].requires_grad is False
