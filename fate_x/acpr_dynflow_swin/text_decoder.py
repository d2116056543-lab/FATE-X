from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Optional
from types import SimpleNamespace

from src.layers.bert import BertForImageCaptioning
from src.modeling.load_bert import get_bert_model

from .types import DynFlowSwinTextOutput


def _masked_ce(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    valid = mask.bool()
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid].long())


def _gather_masked_logits(sequence_logits: Tensor, masked_pos: Tensor) -> Tensor:
    max_index = sequence_logits.shape[1] - 1
    positions = masked_pos.long().clamp_min(0).clamp_max(max_index)
    gather_index = positions.unsqueeze(-1).expand(-1, -1, sequence_logits.shape[-1])
    return sequence_logits.gather(dim=1, index=gather_index)


def _binary_masked_ce(sequence_logits: Tensor, labels: Tensor, mask: Tensor, offsets: Optional[Tensor] = None) -> Tensor:
    losses = []
    for batch_index in range(sequence_logits.shape[0]):
        valid_positions = mask[batch_index].bool()
        count = int(valid_positions.sum().detach().cpu())
        if count <= 0:
            continue
        offset = int(offsets[batch_index].detach().cpu()) if offsets is not None else 0
        target = labels[batch_index, offset : offset + count].long()
        losses.append(F.cross_entropy(sequence_logits[batch_index, valid_positions], target))
    if not losses:
        return sequence_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _split_flat_caption_losses(
    flat_logits: Tensor,
    masked_ids: Tensor,
    masked_pos: Tensor,
    token_type_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    selected_types = token_type_ids[:, : masked_pos.shape[1]][masked_pos.bool()]
    if masked_ids.shape == masked_pos.shape:
        selected_ids = masked_ids[masked_pos.bool()]
    else:
        selected_ids = masked_ids.reshape(-1)
    selected_ids = selected_ids[selected_ids.ge(0)]
    count = min(flat_logits.shape[0], selected_ids.shape[0], selected_types.shape[0])
    logits = flat_logits[:count]
    labels = selected_ids[:count].to(logits.device).long()
    types = selected_types[:count].to(logits.device)

    def loss_for(mask: Tensor) -> Tensor:
        if not bool(mask.any()):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[mask].float(), labels[mask])

    return loss_for(types.eq(0)), loss_for(types.ne(0))


class DynFlowSwinTextDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 30522,
        factor_dim: int = 256,
        bert_captioner: BertForImageCaptioning | None = None,
        tokenizer=None,
    ):
        super().__init__()
        self.bert_captioner = bert_captioner
        self.tokenizer = tokenizer
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.factor_proj = nn.Linear(factor_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.factor_attention = nn.Linear(hidden_dim, 13)

    @staticmethod
    def _caption_attention(input_ids: Tensor, img_feats: Tensor, attention_mask: Tensor | None) -> Tensor:
        total = input_ids.shape[1] + img_feats.shape[1]
        if attention_mask is not None and attention_mask.ndim == 3 and attention_mask.shape[-1] == total:
            return attention_mask
        return torch.ones(
            input_ids.shape[0],
            total,
            total,
            device=input_ids.device,
            dtype=torch.float32,
        )

    def generate_text(
        self,
        input_ids: Tensor,
        img_feats: Tensor,
        attention_mask: Tensor | None,
        masked_pos: Tensor,
        token_type_ids: Tensor,
        use_sep_cap: bool = True,
        max_length: int = 30,
    ) -> tuple[Tensor, Tensor]:
        if self.bert_captioner is None:
            raise RuntimeError("ADAPT BertForImageCaptioning generation requires bert_captioner")
        outputs = self.bert_captioner.generate(
            input_ids=input_ids,
            img_feats=img_feats,
            attention_mask=self._caption_attention(input_ids, img_feats, attention_mask),
            masked_pos=masked_pos,
            token_type_ids=token_type_ids,
            max_length=max_length,
            use_sep_cap=use_sep_cap,
            do_sample=False,
            bos_token_id=101,
            pad_token_id=0,
            eos_token_ids=[102],
            mask_token_id=103,
            num_beams=1,
            num_return_sequences=1,
            num_keep_best=1,
            is_decode=True,
        )
        sequences = outputs[0]
        if sequences.ndim == 3:
            sequences = sequences[:, 0]
        split = max_length // 2
        return sequences[:, :split], sequences[:, split:max_length]

    def decode_token_ids(self, token_ids: Tensor) -> list[str]:
        if self.tokenizer is None:
            return [" ".join(str(int(token)) for token in row if int(token) > 0) for row in token_ids]
        return [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True).strip()
            for row in token_ids.detach().cpu()
        ]

    def forward(
        self,
        input_ids: Tensor,
        masked_pos: Tensor,
        masked_ids: Tensor,
        text_hidden: Tensor,
        factor_tokens: Tensor,
        token_type_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        visual_tokens: Tensor | None = None,
    ) -> DynFlowSwinTextOutput:
        hidden = self.token_embed(input_ids.clamp_min(0)) + text_hidden
        factor_context = self.factor_proj(factor_tokens.mean(dim=1))
        hidden = hidden + factor_context.mean(dim=1, keepdim=True)
        logits = self.lm_head(hidden)
        if self.bert_captioner is not None and visual_tokens is not None and token_type_ids is not None:
            caption_attention = self._caption_attention(input_ids, visual_tokens, attention_mask)
            _, flat_logits = self.bert_captioner(
                input_ids=input_ids,
                img_feats=visual_tokens,
                attention_mask=caption_attention,
                masked_pos=masked_pos,
                masked_ids=masked_ids,
                token_type_ids=token_type_ids,
                is_training=True,
            )
            action_loss, explanation_loss = _split_flat_caption_losses(
                flat_logits, masked_ids, masked_pos, token_type_ids
            )
        elif masked_pos.shape[1] == input_ids.shape[1] and int(masked_pos.max().detach().cpu()) <= 1:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
            valid_mask = masked_pos.bool()
            action_mask = valid_mask & positions.lt(15)
            explanation_mask = valid_mask & positions.ge(15)
            action_offsets = action_mask.sum(dim=1)
            action_loss = _binary_masked_ce(logits, masked_ids, action_mask)
            explanation_loss = _binary_masked_ce(logits, masked_ids, explanation_mask, offsets=action_offsets)
        else:
            masked_logits = _gather_masked_logits(logits, masked_pos)
            valid_mask = masked_ids.ge(0)
            action_mask = valid_mask & masked_pos.lt(15)
            explanation_mask = valid_mask & masked_pos.ge(15)
            action_loss = _masked_ce(masked_logits, masked_ids, action_mask)
            explanation_loss = _masked_ce(masked_logits, masked_ids, explanation_mask)
        attn = torch.softmax(self.factor_attention(hidden), dim=-1)
        return DynFlowSwinTextOutput(
            total_mlm_loss=action_loss + explanation_loss,
            action_loss=action_loss,
            explanation_loss=explanation_loss,
            action_logits=logits[:, :15],
            explanation_logits=logits[:, 15:30],
            action_to_factor_attention=attn[:, :15].mean(dim=1),
            explanation_to_factor_attention=attn[:, 15:30].mean(dim=1),
            generated_action=[],
            generated_explanation=[],
        )


def build_generic_adapt_captioner(bert_dir: str, img_feature_dim: int = 768):
    args = SimpleNamespace(
        config_name="",
        model_name_or_path=bert_dir,
        tokenizer_name="",
        do_lower_case=True,
        drop_out=0.1,
        tie_weights=True,
        freeze_embedding=False,
        label_smoothing=0.0,
        drop_worst_ratio=0.0,
        drop_worst_after=0,
        img_feature_dim=img_feature_dim,
        num_hidden_layers=-1,
        hidden_size=-1,
        num_attention_heads=-1,
        intermediate_size=-1,
        load_partial_weights=True,
    )
    captioner, _, tokenizer = get_bert_model(args)
    return captioner, tokenizer
