import torch

from fate_x.losses.explanation_scst import sentence_cider_reward, self_critical_explanation_loss


def test_scst_uses_sample_logprobs_and_reward_advantage():
    assert sentence_cider_reward("car slows", "car slows") > sentence_cider_reward("sky blue", "car slows")
    logp = torch.zeros(2, 3, requires_grad=True)
    loss = self_critical_explanation_loss(logp, torch.tensor([1.0, 0.2]), torch.tensor([0.5, 0.5]))
    loss.backward()
    assert logp.grad is not None
