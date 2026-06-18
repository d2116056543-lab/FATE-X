from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FlowTraceLoss(nn.Module):
    DEFAULT_WEIGHTS = {
        "anchor": 0.05,
        "reason_state": 0.05,
        "dynamic_control": 0.05,
        "transport_cycle": 0.01,
        "track_diversity": 0.002,
        "state_diversity": 0.002,
        "state_sparsity": 0.001,
        "preserve_kl": 0.02,
        "intervention": 0.0,
    }

    @classmethod
    def available_components(cls) -> tuple[str, ...]:
        return tuple(cls.DEFAULT_WEIGHTS.keys())

    def __init__(self, state_dim: int = 256, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        merged = dict(self.DEFAULT_WEIGHTS)
        if weights:
            merged.update(weights)
        self.weights = merged
        self.control_from_state = nn.Linear(state_dim, 2)

    def forward(self, bundle, text_loss: Tensor | None = None, control_loss: Tensor | None = None,
                anchor_target: Tensor | None = None, reason_target: Tensor | None = None,
                control_prediction: Tensor | None = None, control_target: Tensor | None = None,
                baseline_logits: Tensor | None = None, pmt_logits: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        device = bundle.state_memory.device
        total = torch.zeros((), device=device)
        logs: dict[str, Tensor] = {}
        if text_loss is not None:
            total = total + text_loss
            logs["text"] = text_loss.detach()
        if control_loss is not None:
            total = total + control_loss
            logs["control"] = control_loss.detach()
        if anchor_target is not None:
            q = bundle.reason_state_distribution.clamp_min(1e-6)
            p = anchor_target.to(q.device).clamp_min(1e-6)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            loss = F.kl_div(q.log(), p, reduction="batchmean")
            total = total + self.weights.get("anchor", 0.05) * loss
            logs["anchor"] = loss.detach()
        if reason_target is not None:
            loss = 1.0 - F.cosine_similarity(bundle.reason_state, reason_target.to(device), dim=-1).mean()
            total = total + self.weights["reason_state"] * loss
            logs["reason_state"] = loss.detach()

        dynamic_control = self._dynamic_control_loss(bundle, control_prediction, control_target)
        total = total + self.weights["dynamic_control"] * dynamic_control
        logs["dynamic_control"] = dynamic_control.detach()

        transport_cycle = self._transport_cycle_loss(bundle.transport_matrices)
        total = total + self.weights["transport_cycle"] * transport_cycle
        logs["transport_cycle"] = transport_cycle.detach()

        track_div = self._diversity_loss(bundle.track_attention.flatten(-2))
        total = total + self.weights["track_diversity"] * track_div
        logs["track_diversity"] = track_div.detach()

        state_div = self._diversity_loss(bundle.state_memory)
        total = total + self.weights["state_diversity"] * state_div
        logs["state_diversity"] = state_div.detach()

        state_sparsity = self._state_sparsity_loss(bundle.state_evidence_maps)
        total = total + self.weights["state_sparsity"] * state_sparsity
        logs["state_sparsity"] = state_sparsity.detach()

        preserve_kl = self._preserve_loss(bundle, baseline_logits, pmt_logits)
        total = total + self.weights["preserve_kl"] * preserve_kl
        logs["preserve_kl"] = preserve_kl.detach()

        intervention, intervention_logs = self._intervention_loss(bundle)
        total = total + self.weights["intervention"] * intervention
        logs["intervention"] = intervention.detach()
        logs.update({k: v.detach() for k, v in intervention_logs.items()})
        return total, logs

    def _dynamic_control_loss(self, bundle, control_prediction: Tensor | None, control_target: Tensor | None) -> Tensor:
        if control_prediction is not None and control_target is not None:
            target = control_target.to(control_prediction.device)
            if target.dim() == 3:
                target = target.permute(0, 2, 1)
            if target.shape[-1] != control_prediction.shape[-1]:
                target = target[..., :control_prediction.shape[-1]]
            return F.smooth_l1_loss(control_prediction.float(), target.float())
        pred = self.control_from_state(bundle.state_memory.mean(dim=1)).float()
        return pred.pow(2).mean()

    @staticmethod
    def _transport_cycle_loss(transport: Tensor) -> Tensor:
        core = transport[..., :-1, :-1].float()
        if core.numel() == 0:
            return transport.new_zeros(())
        row_mass = core.sum(dim=-1)
        col_mass = core.sum(dim=-2)
        return (row_mass - col_mass).abs().mean()

    @staticmethod
    def _diversity_loss(tokens: Tensor) -> Tensor:
        x = tokens.float()
        if x.dim() > 3:
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = F.normalize(x, dim=-1)
        sim = torch.matmul(x, x.transpose(-1, -2)).abs()
        n = sim.shape[-1]
        if n <= 1:
            return sim.new_zeros(())
        eye = torch.eye(n, device=sim.device, dtype=sim.dtype)
        return (sim * (1.0 - eye)).sum() / (sim.shape[0] * n * (n - 1))

    @staticmethod
    def _state_sparsity_loss(maps: Tensor) -> Tensor:
        probs = maps.float().flatten(-2).clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device, dtype=probs.dtype))
        return (entropy / max_entropy.clamp_min(1e-8)).mean()

    @staticmethod
    def _preserve_loss(bundle, baseline_logits: Tensor | None, pmt_logits: Tensor | None) -> Tensor:
        if baseline_logits is not None and pmt_logits is not None:
            base = F.log_softmax(baseline_logits.float(), dim=-1)
            pmt = F.log_softmax(pmt_logits.float(), dim=-1)
            return F.kl_div(pmt, base.exp(), reduction="batchmean")
        if bundle.pmt_delta is None:
            return bundle.state_memory.new_zeros(())
        return bundle.pmt_delta.float().pow(2).mean()

    @staticmethod
    def _intervention_loss(bundle) -> tuple[Tensor, dict[str, Tensor]]:
        scores = bundle.state_scores.float()
        memory = bundle.state_memory.float()
        if scores.dim() == 3:
            scores = scores.mean(dim=1)
        if scores.numel() == 0 or memory.numel() == 0 or scores.shape[:2] != memory.shape[:2]:
            zero = memory.new_zeros(())
            return zero, {"intervention_available": zero}

        probs = torch.softmax(scores, dim=-1)
        contributions = probs.unsqueeze(-1) * memory
        full_state = contributions.sum(dim=1)
        state_off = full_state.unsqueeze(1) - contributions
        state_off_delta = (full_state.unsqueeze(1) - state_off).norm(dim=-1)

        high_delta = state_off_delta.max(dim=1).values
        equal_mass_delta = state_off_delta.mean(dim=1)
        rank_margin = 0.01
        rank_loss = F.relu(equal_mass_delta - high_delta + rank_margin).mean()
        magnitude = high_delta.mean()
        loss = rank_loss + 0.01 * magnitude
        return loss, {
            "intervention_available": memory.new_ones(()),
            "intervention_state_off_delta": high_delta.mean(),
            "intervention_equal_mass_delta": equal_mass_delta.mean(),
        }
