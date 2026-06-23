import torch
import torch.nn.functional as F

from fate_x.acpr_dynflow.text_decoder import DynFlowTextDecoder


def test_binary_masked_pos_unpacking_splits_action_and_explanation_loss():
    decoder = DynFlowTextDecoder(text_dim=8, factor_dim=6, vocab_size=11)
    logits = torch.randn(1, 6, 11)
    masked_pos = torch.tensor([[0, 1, 0, 1, 1, 0]])
    masked_ids = torch.tensor([[2, 3, 4, -1]])

    action_loss = decoder._masked_position_loss(logits, masked_pos, masked_ids, 0, 3)
    explanation_loss = decoder._masked_position_loss(logits, masked_pos, masked_ids, 3, 6)

    expected_action = F.cross_entropy(logits[0, [1]], torch.tensor([2]))
    expected_explanation = F.cross_entropy(logits[0, [3, 4]], torch.tensor([3, 4]))

    assert torch.allclose(action_loss, expected_action)
    assert torch.allclose(explanation_loss, expected_explanation)
    assert float(explanation_loss) > 0.0
