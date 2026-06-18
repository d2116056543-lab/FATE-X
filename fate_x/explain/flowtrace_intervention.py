from __future__ import annotations

import torch


class FlowTraceInterventionRunner:
    def apply(self, bundle, intervention: dict | None):
        if intervention is None:
            return bundle
        if intervention.get("type") == "state_off":
            idx = int(intervention["state_idx"])
            bundle.state_memory = bundle.state_memory.clone()
            bundle.state_memory[:, idx] = 0
            return bundle
        if intervention.get("type") == "evidence_tube_off":
            idx = int(intervention.get("state_idx", 0))
            bundle.state_evidence_maps = bundle.state_evidence_maps.clone()
            bundle.state_evidence_maps[:, :, idx] = 0
            return bundle
        if intervention.get("type") == "random_equal_mass":
            seed = int(intervention.get("seed", 0))
            gen = torch.Generator(device=bundle.state_memory.device).manual_seed(seed)
            idx = torch.randint(0, bundle.state_memory.shape[1], (1,), generator=gen, device=bundle.state_memory.device).item()
            bundle.state_memory = bundle.state_memory.clone()
            bundle.state_memory[:, idx] = 0
            return bundle
        if intervention.get("type") == "temporal_shuffle":
            seed = int(intervention.get("seed", 0))
            gen = torch.Generator(device=bundle.state_tokens_temporal.device).manual_seed(seed)
            order = torch.randperm(bundle.state_tokens_temporal.shape[1], generator=gen, device=bundle.state_tokens_temporal.device)
            bundle.state_tokens_temporal = bundle.state_tokens_temporal[:, order]
            return bundle
        if intervention.get("type") == "temporal_reverse":
            bundle.state_tokens_temporal = torch.flip(bundle.state_tokens_temporal, dims=[1])
            return bundle
        raise ValueError(f"Unsupported intervention: {intervention}")
