
import torch


def test_text_decoder_binary_mask_and_separate_segments():
    from fate_x.acpr_dynflow_swin.text_decoder import DynFlowSwinTextDecoder

    decoder = DynFlowSwinTextDecoder(hidden_dim=32, vocab_size=128, factor_dim=16)
    hidden = torch.randn(2, 30, 32)
    factor_tokens = torch.randn(2, 32, 13, 16)
    input_ids = torch.randint(0, 128, (2, 30))
    masked_pos = torch.zeros(2, 30, dtype=torch.long)
    masked_pos[:, :15] = 1
    masked_pos[:, 15:] = 1
    masked_ids = torch.randint(0, 128, (2, 30))
    out = decoder(input_ids, masked_pos, masked_ids, hidden, factor_tokens)
    assert torch.isfinite(out.action_loss)
    assert torch.isfinite(out.explanation_loss)
    assert out.action_loss.item() > 0
    assert out.explanation_loss.item() > 0
    assert out.action_logits.shape[:2] == (2, 15)
    assert out.explanation_logits.shape[:2] == (2, 15)
