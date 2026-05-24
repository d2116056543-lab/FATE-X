import torch

from fate_x.explain.video_token_attribution import gradient_x_activation, normalize_token_scores


def test_gradient_x_activation_and_normalize():
    tokens = torch.ones(2, 4, 3)
    grads = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    scores = gradient_x_activation(tokens, grads)
    assert scores.shape == (2, 4)
    norm = normalize_token_scores(scores)
    assert torch.all(norm >= 0)
    assert torch.all(norm <= 1)