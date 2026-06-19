from fate_x.acpr_flow.model import ACPRFlowModel
from fate_x.engine.train_acpr_flowcal_pp import build_acpr_optimizer_groups


def test_optimizer_groups_cover_every_trainable_parameter_exactly_once():
    model = ACPRFlowModel()
    groups, manifest = build_acpr_optimizer_groups(model)
    group_names = [name for g in groups for name in g["names"]]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert sorted(group_names) == sorted(trainable_names)
    assert len(group_names) == len(set(group_names))
    assert manifest["predicate_field.queries"] == "predicate_field"
