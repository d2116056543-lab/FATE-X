from fate_x.acpr_dynflow.model import ACPRDynFlowModel

def test_trainable_params_exist():
    m=ACPRDynFlowModel()
    assert any(p.requires_grad for p in m.parameters())
    assert all(not p.requires_grad for p in m.backbone.stage0.parameters())

