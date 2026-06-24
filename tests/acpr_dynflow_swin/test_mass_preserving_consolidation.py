
import torch


def test_mass_preserving_consolidation_identity():
    from fate_x.acpr_dynflow_swin.semantic_token_consolidator import SemanticTokenConsolidator

    module = SemanticTokenConsolidator(input_dim=24, output_dim=24)
    tokens = torch.randn(2, 16, 49, 24)
    out = module(tokens)
    assert out.assignment.shape == (2, 16, 49, 5)
    assert torch.allclose(out.assignment.sum(-1), torch.ones(2, 16, 49), atol=1e-6)
    assert float(out.conservation_error.max()) < 1e-5
    weighted = (out.assignment.unsqueeze(-1) * out.tokens.unsqueeze(2)).sum(dim=3).sum(dim=2)
    assert torch.allclose(weighted, tokens.sum(dim=2), atol=1e-4)


def test_mass_conservation_is_stable_for_bf16_inputs():
    from fate_x.acpr_dynflow_swin.semantic_token_consolidator import SemanticTokenConsolidator

    module = SemanticTokenConsolidator(input_dim=24, output_dim=24).to(dtype=torch.bfloat16)
    tokens = torch.randn(1, 16, 49, 24, dtype=torch.bfloat16)
    out = module(tokens)
    assert float(out.conservation_error.max()) <= 1e-3
