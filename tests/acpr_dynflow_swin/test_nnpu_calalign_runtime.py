import torch

from fate_x.acpr_dynflow_swin.nnpu_calalign import (
    OnlineCalAlign,
    PredicateRuleLabeler,
    nnpu_loss,
)
from fate_x.acpr_dynflow_swin.predicate_ontology import EXACT_32_PREDICATES


def test_rules_cover_all_predicates_and_create_three_way_labels():
    labeler = PredicateRuleLabeler.from_yaml("configs/acpr_dynflow_swin_text_rules.yaml")
    assert tuple(labeler.rules) == EXACT_32_PREDICATES

    labels = labeler.label(
        ["The traffic light is red and cars ahead stopped.", "The road is clear with no pedestrian."]
    )
    assert labels.positive.sum().item() > 0
    assert labels.reliable_negative.sum().item() > 0
    assert labels.unlabeled.sum().item() > 0
    assert torch.all(labels.positive + labels.reliable_negative + labels.unlabeled <= 1)


def test_unknown_only_is_not_ordinary_negative_bce():
    logits = torch.zeros(1, 32, requires_grad=True)
    labeler = PredicateRuleLabeler.from_yaml("configs/acpr_dynflow_swin_text_rules.yaml")
    labels = labeler.label(["unknown unrelated words"])
    loss = nnpu_loss(logits, labels, torch.full((32,), 0.1))
    loss.backward()

    assert labels.positive.sum().item() == 0
    assert labels.reliable_negative.sum().item() == 0
    assert labels.unlabeled.sum().item() == 32
    assert torch.isfinite(loss)


def test_calalign_updates_only_in_train_and_roundtrips_state():
    cal = OnlineCalAlign(32)
    logits = torch.randn(4, 32)
    labeler = PredicateRuleLabeler.from_yaml("configs/acpr_dynflow_swin_text_rules.yaml")
    labels = labeler.label(
        ["red light", "green light", "cars ahead stopped", "road is clear"]
    )
    before = {k: v.clone() for k, v in cal.state_dict().items()}
    cal.update(logits, labels, training=False)
    assert all(torch.equal(before[k], v) for k, v in cal.state_dict().items())
    cal.update(logits, labels, training=True)
    assert any(not torch.equal(before[k], v) for k, v in cal.state_dict().items())

    restored = OnlineCalAlign(32)
    restored.load_state_dict(cal.state_dict())
    assert all(torch.equal(cal.state_dict()[k], restored.state_dict()[k]) for k in cal.state_dict())


def test_calalign_keeps_fp32_state_with_bf16_logits():
    cal = OnlineCalAlign(32)
    logits = torch.randn(2, 32, dtype=torch.bfloat16)
    labeler = PredicateRuleLabeler.from_yaml("configs/acpr_dynflow_swin_text_rules.yaml")
    labels = labeler.label(["red light", "road is clear"])
    cal.update(logits, labels, training=True)
    assert cal.prior.dtype == torch.float32
    assert cal.temperature.dtype == torch.float32
