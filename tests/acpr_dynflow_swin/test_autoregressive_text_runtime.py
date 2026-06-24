import torch
from torch import nn

from fate_x.acpr_dynflow_swin.text_decoder import DynFlowSwinTextDecoder


class FakeCaptioner(nn.Module):
    def __init__(self, vocab_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.decode_calls = 0
        self.train_calls = 0

    def forward(self, **kwargs):
        self.train_calls += 1
        count = int(kwargs["masked_pos"].bool().sum())
        return torch.tensor(0.0), torch.randn(count, self.vocab_size)

    def generate(self, **kwargs):
        self.decode_calls += 1
        batch = kwargs["img_feats"].shape[0]
        ids = torch.arange(30).view(1, 1, 30).expand(batch, 1, 30)
        return ids, torch.zeros(batch, 1)


def test_decoder_uses_captioner_for_training_and_autoregressive_generation():
    captioner = FakeCaptioner()
    decoder = DynFlowSwinTextDecoder(
        hidden_dim=16,
        vocab_size=32,
        factor_dim=8,
        bert_captioner=captioner,
    )
    input_ids = torch.zeros(2, 30, dtype=torch.long)
    masked_pos = torch.ones(2, 30, dtype=torch.long)
    masked_ids = torch.randint(0, 32, (2, 30))
    token_types = torch.cat(
        [torch.zeros(2, 15, dtype=torch.long), torch.ones(2, 15, dtype=torch.long)], dim=1
    )
    visual_tokens = torch.randn(2, 106, 16)
    factor_tokens = torch.randn(2, 32, 13, 8)
    output = decoder(
        input_ids,
        masked_pos,
        masked_ids,
        torch.zeros(2, 30, 16),
        factor_tokens,
        token_type_ids=token_types,
        visual_tokens=visual_tokens,
    )
    generated_action, generated_explanation = decoder.generate_text(
        input_ids=input_ids,
        attention_mask=torch.ones(2, 30),
        masked_pos=masked_pos,
        token_type_ids=token_types,
        img_feats=visual_tokens,
    )

    assert captioner.train_calls == 1
    assert captioner.decode_calls == 1
    assert output.action_loss.item() > 0
    assert output.explanation_loss.item() > 0
    assert generated_action.shape == (2, 15)
    assert generated_explanation.shape == (2, 15)
