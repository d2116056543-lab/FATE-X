from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .activations import entmax15


def scaled_gradient(memory: Tensor, scale: float) -> Tensor:
    return memory.detach() + float(scale) * (memory - memory.detach())


def match_beam_batch(tensor: Tensor | None, target_batch: int, name: str) -> Tensor | None:
    if tensor is None or tensor.shape[0] == target_batch:
        return tensor
    source_batch = int(tensor.shape[0])
    if source_batch > 0 and target_batch % source_batch == 0:
        return tensor.repeat_interleave(target_batch // source_batch, dim=0)
    raise RuntimeError(
        f"{name} batch size {source_batch} cannot be aligned to hidden batch size {target_batch}"
    )


class TemporalSECA(nn.Module):
    def __init__(self, hidden_dim: int = 768, max_action_scale: float = 0.15, max_explanation_scale: float = 0.30,
                 action_grad_scale: float = 0.25, explanation_grad_scale: float = 1.0) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.out.weight)
        self.gamma_action_raw = nn.Parameter(torch.tensor(0.0))
        self.gamma_explanation_raw = nn.Parameter(torch.tensor(0.0))
        self.max_action_scale = max_action_scale
        self.max_explanation_scale = max_explanation_scale
        self.action_grad_scale = action_grad_scale
        self.explanation_grad_scale = explanation_grad_scale

    def forward(self, hidden: Tensor, reason_memory: Tensor, token_type_ids: Tensor | None = None,
                text_len: int | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        batch_size = int(hidden.shape[0])
        reason_memory = match_beam_batch(reason_memory, batch_size, "reason_memory")
        token_type_ids = match_beam_batch(token_type_ids, batch_size, "token_type_ids")
        if text_len is None:
            text_len = hidden.shape[1]
        text_len = min(int(text_len), int(hidden.shape[1]))
        text = hidden[:, :text_len]
        image = hidden[:, text_len:]
        if token_type_ids is None:
            action_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            action_mask = token_type_ids[:, :text_len].eq(0).to(device=hidden.device)
            if action_mask.shape[1] < text.shape[1]:
                pad = torch.ones(
                    action_mask.shape[0],
                    text.shape[1] - action_mask.shape[1],
                    dtype=torch.bool,
                    device=hidden.device,
                )
                action_mask = torch.cat([action_mask, pad], dim=1)
            elif action_mask.shape[1] > text.shape[1]:
                action_mask = action_mask[:, : text.shape[1]]
        memory_action = scaled_gradient(reason_memory, self.action_grad_scale)
        memory_exp = scaled_gradient(reason_memory, self.explanation_grad_scale)
        q = self.q(text)
        ka, va = self.k(memory_action), self.v(memory_action)
        ke, ve = self.k(memory_exp), self.v(memory_exp)
        attn_a = entmax15(torch.matmul(q, ka.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
        attn_e = entmax15(torch.matmul(q, ke.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
        ctx = torch.where(action_mask.unsqueeze(-1), torch.matmul(attn_a, va), torch.matmul(attn_e, ve))
        delta = self.out(ctx).tanh()
        scale_a = self.max_action_scale * torch.tanh(self.gamma_action_raw)
        scale_e = self.max_explanation_scale * torch.tanh(self.gamma_explanation_raw)
        scale = torch.where(action_mask.unsqueeze(-1), scale_a, scale_e)
        text_out = text + scale * delta
        out = torch.cat([text_out, image], dim=1)
        attn = torch.where(action_mask.unsqueeze(-1), attn_a, attn_e)
        if image.numel() == 0:
            image_diff = hidden.new_zeros(())
        else:
            image_diff = (out[:, text_len:] - image).abs().max()
        return out, {"token_reason_attention": attn, "token_delta": scale * delta, "image_hidden_max_diff": image_diff}
