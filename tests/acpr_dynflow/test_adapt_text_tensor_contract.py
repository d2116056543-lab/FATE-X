def test_required_contract_file_present():
    assert True

def test_text_decoder_accepts_adapt_masked_token_contract():
    import torch

    from fate_x.acpr_dynflow.text_decoder import DynFlowTextDecoder
    from fate_x.acpr_dynflow.types import DecisionLedger, TrafficFlowState

    batch = 2
    text_len = 30
    steps = 32
    factors = 13
    dim = 256
    decoder = DynFlowTextDecoder(text_dim=32, factor_dim=dim, vocab_size=101)
    input_ids = torch.randint(0, 101, (batch, text_len))
    masked_pos = torch.tensor([[0, 3, 14, 15, 20, 29] + [0] * 39, [2, 7, 12, 17, 22, 28] + [0] * 39])
    masked_ids = torch.randint(0, 101, (batch, 45))
    masked_ids[:, 6:] = -1
    flow = TrafficFlowState(
        factor_names=tuple(f"factor_{i}" for i in range(factors)),
        factor_tokens=torch.randn(batch, steps, factors, dim),
        factor_logits=torch.randn(batch, steps, factors),
        factor_probs=torch.softmax(torch.randn(batch, steps, factors), dim=-1),
        lateral_bias=torch.randn(batch, steps, 3),
        factor_to_predicate=torch.softmax(torch.randn(batch, steps, factors, 32), dim=-1),
        evidence_maps=torch.randn(batch, steps, factors, 7, 7),
        response_lag_weights=torch.softmax(torch.randn(batch, steps, factors, 4), dim=-1),
        lag_aligned_tokens=torch.randn(batch, steps, factors, dim),
        lineage=[],
    )
    ledger = DecisionLedger(
        signal_names=("course", "speed"),
        global_prediction_normalized=torch.randn(batch, steps, 2),
        factor_contributions_normalized=torch.randn(batch, steps, factors, 2),
        final_prediction_normalized=torch.randn(batch, steps, 2),
        global_prediction_raw=torch.randn(batch, steps, 2),
        factor_contributions_raw=torch.randn(batch, steps, factors, 2),
        final_prediction_raw=torch.randn(batch, steps, 2),
        speed_factor_attention=torch.softmax(torch.randn(batch, steps, factors), dim=-1),
        course_factor_attention=torch.softmax(torch.randn(batch, steps, factors), dim=-1),
    )

    out = decoder(input_ids, masked_pos, masked_ids, flow, ledger)

    assert out.action_logits.shape[:2] == (batch, text_len)
    assert out.explanation_logits.shape[:2] == (batch, text_len)
    assert torch.isfinite(out.action_loss)
    assert torch.isfinite(out.explanation_loss)

