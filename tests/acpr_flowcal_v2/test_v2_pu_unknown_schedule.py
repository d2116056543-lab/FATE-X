import torch

from fate_x.acpr_flow_v2.pu_targets import FreeTextPUTargetBuilderV2, positive_unlabeled_loss_v2


def test_pu_unknowns_are_low_weight_not_hard_negative():
    builder = FreeTextPUTargetBuilderV2(names=["slow", "stop"], unknown_weight=0.01)
    batch = builder.build(["slow"], ["traffic"])
    loss = positive_unlabeled_loss_v2(torch.zeros_like(batch.targets), batch.targets, batch.known_mask, batch.unknown_weight)
    assert torch.isfinite(loss)
    assert batch.unknown_weight == 0.01
