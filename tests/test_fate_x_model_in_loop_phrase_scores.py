from __future__ import annotations

from fate_x.engine.generate_decoder_phrase_scores_from_model import collect_phrase_scores_from_model


class MockPhraseModel:
    def generate_with_logprobs(self, sample, mask=None):
        text = "A pedestrian is crossing."
        tokens = ["A", "pedestrian", "is", "crossing", "."]
        if mask == "topk":
            logprobs = [-0.1, -1.2, -0.2, -1.4, -0.1]
        elif mask == "evidence_only":
            logprobs = [-0.1, -0.1, -0.2, -0.2, -0.1]
        elif mask == "random":
            logprobs = [-0.1, -0.5, -0.2, -0.6, -0.1]
        else:
            logprobs = [-0.1, -0.2, -0.2, -0.3, -0.1]
        return {"prediction": text, "tokens": tokens, "token_logprobs": logprobs}


def test_model_in_loop_mock_phrase_scores_drop_under_topk_mask():
    rows = collect_phrase_scores_from_model(MockPhraseModel(), [{"id": "s0"}], topk_ratio=0.1)
    assert rows[0]["phrase_hit_count"] >= 1
    first = rows[0]["phrase_faithfulness"][0]
    assert first["topk_masked_score"] < first["original_score"]
    assert "evidence_only_score" in first
