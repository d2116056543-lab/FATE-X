import torch

from fate_x.acpr_flow.flow_factor_composer import FlowFactorComposer
from fate_x.acpr_flow.region_priors import FLOW_FACTOR_NAMES


def test_flow_factor_composer_has_exact_13_factorized_outputs():
    composer = FlowFactorComposer(state_dim=16)
    desc = torch.randn(2, 32, 16, requires_grad=True)
    maps = torch.rand(2, 4, 32, 3, 3)
    out = composer(desc, maps)
    assert out["flow_factor_names"] == FLOW_FACTOR_NAMES
    assert out["flow_tokens"].shape == (2, 13, 16)
    assert out["flow_logits"].shape == (2, 13)
    assert out["flow_to_predicate_attention"].shape == (2, 13, 32)
    assert out["flow_evidence_maps"].shape == (2, 4, 13, 3, 3)
    out["flow_logits"].sum().backward()
    assert composer.queries.grad.abs().sum() > 0
