
import torch


def test_exact_decision_ledger_identity_and_safe_loss_direction():
    from fate_x.acpr_dynflow_swin.decision_ledger import ExactDecisionLedgerHead

    head = ExactDecisionLedgerHead(dim=16, factor_count=13, signal_names=("course", "speed"))
    global_pred = torch.randn(2, 32, 2)
    factors = torch.randn(2, 32, 13, 16)
    ledger = head(global_pred, factors)
    reconstructed = ledger.global_prediction_normalized + ledger.gated_factor_contributions_normalized.sum(dim=2)
    assert torch.allclose(reconstructed, ledger.final_prediction_normalized, atol=1e-6)
    assert ledger.raw_factor_contributions_normalized.shape[-1] == 2
    assert ledger.speed_factor_attention.shape == (2, 32, 13)
    assert ledger.course_factor_attention.shape == (2, 32, 13)


def test_speed_and_course_use_independent_factor_readers():
    from fate_x.acpr_dynflow_swin.decision_ledger import ExactDecisionLedgerHead

    head = ExactDecisionLedgerHead(dim=8, factor_count=13, signal_names=("course", "speed"))
    global_pred = torch.zeros(1, 4, 2)
    factors = torch.randn(1, 4, 13, 8)
    before = head(global_pred, factors).raw_factor_contributions_normalized
    with torch.no_grad():
        head.speed_reader.weight.add_(1.0)
    after = head(global_pred, factors).raw_factor_contributions_normalized
    assert torch.allclose(before[..., 0], after[..., 0])
    assert not torch.allclose(before[..., 1], after[..., 1])
