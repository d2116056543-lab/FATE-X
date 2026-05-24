import torch

from fate_x.explain.phrase_counterfactual import mask_video_tokens, recover_original_token_scores, summarize_phrase_scores, topk_token_mask


def test_phrase_counterfactual_token_masking_and_recovery():
    reduced_scores = torch.tensor([[1.0, 0.0]])
    provenance = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]])
    scores = recover_original_token_scores(reduced_scores, provenance)
    assert scores.shape == (1, 3)
    mask = topk_token_mask(scores, fraction=1 / 3)
    x = torch.ones(1, 3, 4)
    masked = mask_video_tokens(x, mask)
    assert masked.shape == x.shape
    assert masked.sum() < x.sum()


def test_phrase_counterfactual_summary_uses_real_perturbation_scores():
    summary = summarize_phrase_scores([
        {"original_score": 0.9, "topk_masked_score": 0.2, "evidence_only_score": 0.7, "random_masked_score": 0.8},
        {"original_score": 0.6, "topk_masked_score": 0.4, "evidence_only_score": 0.5, "random_masked_score": 0.55},
    ])
    assert summary["faithfulness_available"] is True
    assert summary["phrase_deletion_score"] > summary["random_deletion_score"]