import torch


def test_weighted_loss_total_consumes_each_configured_weight_once():
    from fate_x.losses.acpr_dynflow_swin_losses import weighted_loss_total

    raw = {"a": torch.tensor(2.0), "b": torch.tensor(3.0), "disabled": torch.tensor(99.0)}
    total, weighted = weighted_loss_total(raw, {"a": 0.5, "b": 2.0, "disabled": 0.0})
    assert torch.allclose(total, torch.tensor(7.0))
    assert torch.allclose(weighted["a"], torch.tensor(1.0))
    assert torch.allclose(weighted["b"], torch.tensor(6.0))
    assert torch.allclose(weighted["disabled"], torch.tensor(0.0))


def test_pattern_and_traffic_targets_have_formal_shapes():
    from fate_x.losses.acpr_dynflow_swin_losses import (
        pattern_targets_from_predicates,
        traffic_targets_from_predicates,
    )

    probabilities = torch.rand(2, 8, 32)
    pattern = pattern_targets_from_predicates(probabilities)
    traffic = traffic_targets_from_predicates(probabilities)
    assert pattern.shape == (2, 8)
    assert int(pattern.min()) >= 0
    assert int(pattern.max()) < 4
    assert traffic.shape == (2, 8, 13)
    assert bool(((traffic >= 0) & (traffic <= 1)).all())
