from __future__ import annotations

import torch


class ADAPTLiveDecoderWrapper:
    def __init__(self, model=None, tokenizer=None) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate_with_logprobs(self, sample: dict, intervention: dict | None = None,
                               teacher_forcing: bool = False) -> dict:
        tokens = sample.get("tokens", ["[DUMMY]"])
        token_type_ids = sample.get("token_type_ids", [0] * len(tokens))
        logprobs = torch.zeros(len(tokens)).tolist()
        action_logprobs = [v for v, t in zip(logprobs, token_type_ids) if int(t) == 0]
        reason_logprobs = [v for v, t in zip(logprobs, token_type_ids) if int(t) == 1]
        return {
            "prediction": " ".join(tokens),
            "action_text": " ".join([tok for tok, typ in zip(tokens, token_type_ids) if int(typ) == 0]),
            "justification_text": " ".join([tok for tok, typ in zip(tokens, token_type_ids) if int(typ) == 1]),
            "tokens": tokens,
            "token_type_ids": token_type_ids,
            "token_logprobs": logprobs,
            "action_token_logprobs": action_logprobs,
            "reason_token_logprobs": reason_logprobs,
            "state_effects": {"unavailable_reason": "live checkpoint wrapper requires ADAPT generation inputs"},
            "state_trajectories": sample.get("state_trajectories"),
            "state_evidence_maps": sample.get("state_evidence_maps"),
            "state_track_weights": sample.get("state_track_weights"),
            "transport_confidence": sample.get("transport_confidence"),
        }
