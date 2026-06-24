import torch

from fate_x.acpr_dynflow_swin.pattern_lag_traffic_reasoner import PatternLagTrafficReasoner


def _inputs():
    tokens = torch.randn(2, 8, 32, 16)
    evidence = torch.softmax(torch.randn(2, 8, 32, 4, 4).flatten(-2), dim=-1).reshape(2, 8, 32, 4, 4)
    corridor = torch.softmax(torch.randn(2, 8, 32, 3), dim=-1)
    return tokens, evidence, corridor


def test_pattern_branches_reach_factor_tokens():
    model = PatternLagTrafficReasoner(predicate_dim=16, factor_dim=12)
    tokens, evidence, corridor = _inputs()
    baseline = model(tokens, evidence, corridor, target_steps=16)
    with torch.no_grad():
        model.pattern_fusion.weight.zero_()
    changed = model(tokens, evidence, corridor, target_steps=16)
    assert not torch.allclose(baseline.factor_tokens_native, changed.factor_tokens_native)


def test_lag_weights_change_aligned_tokens_and_are_normalized():
    model = PatternLagTrafficReasoner(predicate_dim=16, factor_dim=12)
    tokens, evidence, corridor = _inputs()
    baseline = model(tokens, evidence, corridor, target_steps=16)
    with torch.no_grad():
        model.lag_logits[:, 0] = 10
        model.lag_logits[:, 1:] = -10
    lag_zero = model(tokens, evidence, corridor, target_steps=16)
    assert torch.allclose(
        lag_zero.lag_weights.sum(dim=-1),
        torch.ones_like(lag_zero.lag_weights[..., 0]),
    )
    assert not torch.allclose(baseline.lag_aligned_tokens, lag_zero.lag_aligned_tokens)
