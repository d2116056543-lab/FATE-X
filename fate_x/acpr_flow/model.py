from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .flow_factor_composer import FlowFactorComposer
from .interventions import InterventionSpec, apply_intervention_to_fused_grid
from .local_partial_transport import LocalPartialTransport
from .prefix_future_head import PrefixFutureHead
from .reason_control_adapter import ReasonControlAdapter
from .reason_memory import ReasonMemory
from .temporal_predicate_field import TemporalPredicateEmbeddingField
from .temporal_seca import TemporalSECA
from .types import ACPRFlowBundle, ACPRFlowTrainOutput


@dataclass
class ACPRFlowModelConfig:
    state_dim: int = 256
    text_hidden_dim: int = 768
    num_frames: int = 32
    image_resolution: int = 224
    use_transport: bool = True
    use_flow: bool = True
    use_prefix_future: bool = True


class TinyDirectImageVideoBackbone(nn.Module):
    """Direct-frame backbone used by ACPR tests/smokes when ADAPT Video Swin is unavailable.

    Formal training can inject real Video Swin grids through `precomputed_grids`; this module
    keeps the ACPR path direct-image/no-cache and exposes fine/coarse/fused grids.
    """

    def __init__(self, state_dim: int = 256) -> None:
        super().__init__()
        self.fine = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(64, state_dim, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
        )
        self.coarse = nn.Sequential(
            nn.Conv2d(state_dim, state_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.fuse = nn.LayerNorm(state_dim)

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        if frames.ndim != 5:
            raise ValueError(f"ACPR expects direct frames [B,T,3,H,W], got {tuple(frames.shape)}")
        b, t, c, h, w = frames.shape
        if t != 32 or c != 3:
            raise ValueError(f"ACPR formal path expects [B,32,3,H,W], got {tuple(frames.shape)}")
        x = frames.reshape(b * t, c, h, w)
        fine = self.fine(x).permute(0, 2, 3, 1).reshape(b, t, h // 16, w // 16, -1)
        coarse_2d = self.coarse(fine.reshape(b * t, h // 16, w // 16, -1).permute(0, 3, 1, 2))
        coarse = coarse_2d.permute(0, 2, 3, 1).reshape(b, t, h // 32, w // 32, -1)
        coarse_up = F.interpolate(
            coarse_2d,
            size=fine.shape[2:4],
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape_as(fine)
        fused = self.fuse(fine + coarse_up)
        return {"fine_grid": fine, "coarse_grid": coarse, "fused_grid": fused}


class ACPRFlowModel(nn.Module):
    def __init__(self, config: ACPRFlowModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ACPRFlowModelConfig()
        self.backbone = TinyDirectImageVideoBackbone(self.config.state_dim)
        self.transport = LocalPartialTransport(self.config.state_dim)
        self.predicate_field = TemporalPredicateEmbeddingField(self.config.state_dim)
        self.flow_composer = FlowFactorComposer(self.config.state_dim)
        self.reason_memory = ReasonMemory(self.config.state_dim, self.config.text_hidden_dim)
        self.temporal_seca = TemporalSECA(self.config.text_hidden_dim)
        self.reason_control_adapter = ReasonControlAdapter(self.config.text_hidden_dim, signals=2)
        self.prefix_future_head = PrefixFutureHead(self.config.text_hidden_dim, signals=2)
        self.reason_to_action = nn.Linear(self.config.text_hidden_dim, 30522)
        self.reason_to_explanation = nn.Linear(self.config.text_hidden_dim, 30522)
        self.base_lm = nn.Linear(self.config.text_hidden_dim, 30522)
        self.control_base = nn.Linear(self.config.text_hidden_dim, 2)
        self.control_hidden = nn.Linear(self.config.text_hidden_dim, self.config.text_hidden_dim)

    def build_bundle(self, frames: Tensor, intervention: InterventionSpec | None = None,
                     precomputed_grids: dict[str, Tensor] | None = None) -> ACPRFlowBundle:
        grids = precomputed_grids or self.backbone(frames)
        fused = grids["fused_grid"]
        if intervention is not None and intervention.kind != "none":
            fused = apply_intervention_to_fused_grid(fused, None, intervention)
        transport = self.transport(fused) if self.config.use_transport else {
            "local_transport_probs": fused.new_zeros(fused.shape[0], fused.shape[1] - 1, fused.shape[2] * fused.shape[3], 26),
            "dustbin_prob": fused.new_zeros(fused.shape[0], fused.shape[1] - 1, fused.shape[2] * fused.shape[3]),
            "camera_shift": fused.new_zeros(fused.shape[0], fused.shape[1] - 1, 2),
        }
        predicates = self.predicate_field(fused, transport)
        flows = self.flow_composer(predicates["descriptor"], predicates["attention"]) if self.config.use_flow else {
            "flow_tokens": fused.new_zeros(fused.shape[0], 13, self.config.state_dim),
            "flow_logits": fused.new_zeros(fused.shape[0], 13),
            "flow_probs": fused.new_zeros(fused.shape[0], 13),
            "flow_to_predicate_attention": fused.new_zeros(fused.shape[0], 13, 32),
            "flow_evidence_maps": fused.new_zeros(fused.shape[0], fused.shape[1], 13, fused.shape[2], fused.shape[3]),
        }
        memory = self.reason_memory(
            predicates["descriptor"],
            flows["flow_tokens"],
            flows["flow_to_predicate_attention"],
            acpr_x=not self.config.use_flow,
        )
        return ACPRFlowBundle(
            fine_grid=grids["fine_grid"],
            coarse_grid=grids["coarse_grid"],
            fused_grid=fused,
            camera_shift=transport["camera_shift"],
            local_transport_probs=transport["local_transport_probs"],
            transport_dustbin=transport["dustbin_prob"],
            predicate_attention=predicates["attention"],
            predicate_tokens_temporal=predicates["tokens"],
            predicate_logits_temporal=predicates["presence_logits"],
            predicate_probs_temporal=predicates["presence_probs"],
            predicate_confidence=predicates["trajectory_confidence"],
            predicate_relative_motion=predicates["relative_motion"],
            predicate_descriptor=predicates["descriptor"],
            flow_tokens=flows["flow_tokens"],
            flow_logits=flows["flow_logits"],
            flow_probs=flows["flow_probs"],
            flow_to_predicate_attention=flows["flow_to_predicate_attention"],
            flow_evidence_maps=flows["flow_evidence_maps"],
            local_reason_memory=memory["local_reason_memory"],
            flow_reason_memory=memory["flow_reason_memory"],
            null_reason_memory=memory["null_reason_memory"],
            reason_memory=memory["reason_memory"],
            reason_memory_mask=memory["reason_memory_mask"],
            global_reason_state=memory["global_reason_state"],
            diagnostics={
                "transport_memory_report": self.transport.last_memory_report,
                "predicate_names": predicates["predicate_names"],
                "flow_factor_names": flows.get("flow_factor_names", []),
            },
        )

    def forward(self, frames: Tensor, control_targets: Tensor | None = None,
                intervention: InterventionSpec | None = None) -> ACPRFlowTrainOutput:
        bundle = self.build_bundle(frames, intervention=intervention)
        b = frames.shape[0]
        text_hidden = bundle.global_reason_state.unsqueeze(1).expand(b, 16, -1)
        baseline_masked_logits = self.base_lm(text_hidden)
        enhanced_hidden, seca_info = self.temporal_seca(text_hidden, bundle.reason_memory, text_len=text_hidden.shape[1])
        bundle.token_reason_attention = seca_info["token_reason_attention"]
        bundle.token_delta = seca_info["token_delta"]
        enhanced_masked_logits = self.base_lm(enhanced_hidden)
        control_hidden = self.control_hidden(bundle.global_reason_state).unsqueeze(1).expand(b, 32, -1)
        control_base = self.control_base(control_hidden)
        ctrl = self.reason_control_adapter(control_base, control_hidden, bundle.reason_memory)
        bundle.control_reason_attention = ctrl["control_reason_attention"]
        bundle.control_delta = ctrl["control_delta"]
        control_final = ctrl["control_final_prediction"]
        target_ids = torch.zeros(b, text_hidden.shape[1], dtype=torch.long, device=frames.device)
        action_loss = F.cross_entropy(enhanced_masked_logits.reshape(-1, enhanced_masked_logits.shape[-1]), target_ids.reshape(-1))
        explanation_loss = F.cross_entropy((enhanced_masked_logits + 0.01 * self.reason_to_explanation(enhanced_hidden)).reshape(-1, enhanced_masked_logits.shape[-1]), target_ids.reshape(-1))
        if control_targets is None:
            control_loss = control_final.sum() * 0.0
        else:
            mask = control_targets.ne(-1.0)
            control_loss = ((control_final - control_targets).pow(2) * mask).sum() / mask.sum().clamp_min(1)
        predicate_pu = bundle.predicate_probs_temporal.mean() * 0.0
        flow_pu = bundle.flow_probs.mean() * 0.0
        memory_diversity = bundle.reason_memory.var(dim=1).mean()
        auxiliary_loss = 0.001 * memory_diversity + predicate_pu + flow_pu
        total = action_loss + explanation_loss + control_loss + auxiliary_loss
        return ACPRFlowTrainOutput(
            action_text_loss=action_loss,
            explanation_text_loss=explanation_loss,
            text_loss_total=action_loss + explanation_loss,
            control_loss=control_loss,
            control_base_prediction=control_base,
            control_final_prediction=control_final,
            baseline_masked_logits=baseline_masked_logits,
            enhanced_masked_logits=enhanced_masked_logits,
            auxiliary_loss=auxiliary_loss,
            total_loss=total,
            loss_components={
                "action_text": action_loss,
                "explanation_text": explanation_loss,
                "control": control_loss,
                "predicate_pu": predicate_pu,
                "flow_pu": flow_pu,
                "memory_diversity": memory_diversity,
                "total": total,
            },
            bundle=bundle,
        )
