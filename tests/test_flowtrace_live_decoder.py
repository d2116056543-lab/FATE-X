from fate_x.engine.adapt_live_decoder_wrapper import ADAPTLiveDecoderWrapper


def test_live_decoder_returns_logprob_schema():
    out = ADAPTLiveDecoderWrapper().generate_with_logprobs({"tokens": ["car", "stops", "because", "light"], "token_type_ids": [0,0,1,1]})
    assert "token_logprobs" in out
    assert len(out["action_token_logprobs"]) == 2
    assert len(out["reason_token_logprobs"]) == 2
