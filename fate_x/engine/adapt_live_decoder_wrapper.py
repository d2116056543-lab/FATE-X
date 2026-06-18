from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class ADAPTLiveDecoderWrapper:
    def __init__(self, model=None, tokenizer=None, device: str | torch.device | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else None

    def _decode_tokens(self, token_ids: list[int]) -> list[str]:
        if self.tokenizer is None:
            return [str(x) for x in token_ids]
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def _split_text(self, tokens: list[str], token_type_ids: list[int]) -> tuple[str, str]:
        action = [tok for tok, typ in zip(tokens, token_type_ids) if int(typ) == 0]
        reason = [tok for tok, typ in zip(tokens, token_type_ids) if int(typ) == 1]
        if self.tokenizer is not None:
            return self.tokenizer.convert_tokens_to_string(action), self.tokenizer.convert_tokens_to_string(reason)
        return " ".join(action), " ".join(reason)

    def _apply_intervention(self, sample: dict[str, Any], intervention: dict[str, Any] | None) -> dict[str, Any]:
        if intervention is None:
            return dict(sample)
        edited = dict(sample)
        edited["flowtrace_intervention"] = intervention
        return edited

    def generate_with_logprobs(
        self,
        sample: dict[str, Any],
        intervention: dict[str, Any] | None = None,
        teacher_forcing: bool = False,
    ) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("ADAPTLiveDecoderWrapper requires a loaded ADAPT model for live decoding.")
        model = self.model
        model.eval()
        edited = self._apply_intervention(sample, intervention)
        batch = edited.get("batch")
        if batch is None:
            batch = {k: v for k, v in edited.items() if torch.is_tensor(v)}
        if not isinstance(batch, dict):
            raise ValueError("sample must provide a tensor dict under 'batch' or tensor-valued top-level fields.")
        if self.device is not None:
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = None
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
            if logits is None:
                logits = outputs.get("prediction_scores")
        elif isinstance(outputs, (tuple, list)):
            for value in outputs:
                if torch.is_tensor(value) and value.dim() >= 3:
                    logits = value
                    break
        if logits is None:
            raise RuntimeError("Loaded ADAPT model did not return token logits for live decoding.")

        input_ids = batch.get("input_ids")
        if input_ids is None:
            input_ids = edited.get("input_ids")
        if input_ids is None:
            token_ids = logits.argmax(dim=-1)[0].detach().cpu().tolist()
        else:
            token_ids = input_ids[0].detach().cpu().tolist() if torch.is_tensor(input_ids) else list(input_ids[0])

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = edited.get("token_type_ids")
        if token_type_ids is None:
            token_type_list = [0] * len(token_ids)
        else:
            token_type_list = token_type_ids[0].detach().cpu().tolist() if torch.is_tensor(token_type_ids) else list(token_type_ids[0])

        if teacher_forcing and input_ids is not None:
            target = input_ids[:, : logits.shape[1]].to(logits.device)
            logp = F.log_softmax(logits[:, : target.shape[1]], dim=-1)
            gathered = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0].detach().cpu().tolist()
        else:
            pred = logits.argmax(dim=-1)
            logp = F.log_softmax(logits, dim=-1)
            gathered = logp.gather(-1, pred.unsqueeze(-1)).squeeze(-1)[0].detach().cpu().tolist()
            token_ids = pred[0].detach().cpu().tolist()

        tokens = self._decode_tokens(token_ids)
        action_text, justification_text = self._split_text(tokens, token_type_list)
        action_logprobs = [v for v, t in zip(gathered, token_type_list) if int(t) == 0]
        reason_logprobs = [v for v, t in zip(gathered, token_type_list) if int(t) == 1]
        return {
            "prediction": " ".join(tokens),
            "action_text": action_text,
            "justification_text": justification_text,
            "tokens": tokens,
            "token_type_ids": token_type_list,
            "token_logprobs": gathered,
            "action_token_logprobs": action_logprobs,
            "reason_token_logprobs": reason_logprobs,
            "state_effects": edited.get("state_effects", {}),
            "state_trajectories": edited.get("state_trajectories"),
            "state_evidence_maps": edited.get("state_evidence_maps"),
            "state_track_weights": edited.get("state_track_weights"),
            "transport_confidence": edited.get("transport_confidence"),
        }
