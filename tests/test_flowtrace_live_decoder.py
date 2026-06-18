import pytest
import torch

from fate_x.engine.adapt_live_decoder_wrapper import ADAPTLiveDecoderWrapper


class TinyDecoder(torch.nn.Module):
    def forward(self, **batch):
        vocab = 8
        input_ids = batch["input_ids"]
        logits = torch.full((*input_ids.shape, vocab), -8.0)
        logits.scatter_(-1, input_ids.unsqueeze(-1), 0.0)
        return {"logits": logits}


def test_live_decoder_requires_loaded_model():
    with pytest.raises(ValueError):
        ADAPTLiveDecoderWrapper().generate_with_logprobs({"tokens": ["car"]})


def test_live_decoder_returns_logprob_schema_from_model_logits():
    sample = {
        "batch": {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "token_type_ids": torch.tensor([[0, 0, 1, 1]]),
        }
    }
    out = ADAPTLiveDecoderWrapper(TinyDecoder()).generate_with_logprobs(sample, teacher_forcing=True)
    assert "token_logprobs" in out
    assert len(out["action_token_logprobs"]) == 2
    assert len(out["reason_token_logprobs"]) == 2
    assert all(v <= 0.0 for v in out["token_logprobs"])
