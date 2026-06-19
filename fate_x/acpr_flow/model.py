from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from fate_x.losses.acpr_flowcal_losses import memory_diversity_loss, partial_label_bce, masked_l2_loss

from .adapt_backbone import ADAPTVideoSwinMultiscaleBackbone
from .flow_factor_composer import FlowFactorComposer
from .free_text_partial_targets import FreeTextPartialTargetBuilder
from .interventions import InterventionSpec, apply_intervention_to_fused_grid
from .local_partial_transport import LocalPartialTransport
from .prefix_future_head import PrefixFutureHead
from .reason_control_adapter import ReasonControlAdapter
from .reason_memory import ReasonMemory
from .temporal_predicate_field import TemporalPredicateEmbeddingField
from .temporal_seca import TemporalSECA
from .types import ACPRFlowBatch, ACPRFlowBundle, ACPRFlowTrainOutput


@dataclass
class ACPRFlowModelConfig:
    state_dim: int = 256
    text_hidden_dim: int = 768
    num_frames: int = 32
    image_resolution: int = 224
    formal_backbone: bool = False
    load_pretrained_backbone: bool = True
    video_swin_checkpoint: str | None = None
    fine_stage: int = 2
    coarse_stage: int = 3
    use_transport: bool = True
    use_flow: bool = True
    use_prefix_future: bool = True
    vocab_size: int = 30522
    invalid_control_value: float = -1.0


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
    def __init__(self, config: ACPRFlowModelConfig | None = None, backbone: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config or ACPRFlowModelConfig()
        if backbone is not None:
            self.backbone = backbone
        elif self.config.formal_backbone:
            self.backbone = ADAPTVideoSwinMultiscaleBackbone(
                state_dim=self.config.state_dim,
                image_resolution=self.config.image_resolution,
                fine_stage=self.config.fine_stage,
                coarse_stage=self.config.coarse_stage,
                load_pretrained=self.config.load_pretrained_backbone,
                checkpoint_path=self.config.video_swin_checkpoint,
            )
        else:
            self.backbone = TinyDirectImageVideoBackbone(self.config.state_dim)
        self.transport = LocalPartialTransport(self.config.state_dim)
        self.predicate_field = TemporalPredicateEmbeddingField(self.config.state_dim)
        self.flow_composer = FlowFactorComposer(self.config.state_dim)
        self.reason_memory = ReasonMemory(self.config.state_dim, self.config.text_hidden_dim)
        self.temporal_seca = TemporalSECA(self.config.text_hidden_dim)
        self.reason_control_adapter = ReasonControlAdapter(self.config.text_hidden_dim, signals=2)
        self.prefix_future_head = PrefixFutureHead(self.config.text_hidden_dim, signals=2)
        self.reason_to_action = nn.Linear(self.config.text_hidden_dim, self.config.vocab_size)
        self.reason_to_explanation = nn.Linear(self.config.text_hidden_dim, self.config.vocab_size)
        self.base_lm = nn.Linear(self.config.text_hidden_dim, self.config.vocab_size)
        self.control_base = nn.Linear(self.config.text_hidden_dim, 2)
        self.control_hidden = nn.Linear(self.config.text_hidden_dim, self.config.text_hidden_dim)
        self.text_targets = FreeTextPartialTargetBuilder()

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

    def _resolve_inputs(
        self,
        frames: Tensor | None,
        batch: ACPRFlowBatch | None,
        control_targets: Tensor | None,
        masked_ids: Tensor | None,
        raw_actions: list[str] | None,
        raw_justifications: list[str] | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, list[str], list[str]]:
        if batch is not None:
            frames = batch.frames
            masked_ids = masked_ids if masked_ids is not None else batch.masked_ids
            if control_targets is None and batch.car_info is not None:
                control_targets = batch.car_info.transpose(1, 2).contiguous()
            raw_actions = raw_actions if raw_actions is not None else list(batch.raw_actions)
            raw_justifications = raw_justifications if raw_justifications is not None else list(batch.raw_justifications)
        if frames is None:
            raise ValueError("ACPRFlowModel.forward requires frames or ACPRFlowBatch")
        return frames, control_targets, masked_ids, raw_actions or ["" for _ in range(frames.shape[0])], raw_justifications or ["" for _ in range(frames.shape[0])]

    def _masked_token_loss(self, logits: Tensor, target_ids: Tensor | None) -> Tensor:
        if target_ids is None:
            return logits.mean() * 0.0
        seq = min(logits.shape[1], target_ids.shape[1])
        logits = logits[:, :seq]
        target_ids = target_ids[:, :seq].to(device=logits.device, dtype=torch.long)
        valid = target_ids.ge(0)
        if not bool(valid.any()):
            return logits.mean() * 0.0
        return F.cross_entropy(logits[valid], target_ids[valid])

    def _reason_semantic_loss(self, state: Tensor, target: Tensor | None) -> Tensor:
        if target is None:
            return state.mean() * 0.0
        target = target.to(device=state.device, dtype=state.dtype)
        target = F.normalize(target, dim=-1)
        return (1.0 - F.cosine_similarity(F.normalize(state, dim=-1), target, dim=-1)).mean()

    def _future_control_loss(self, state: Tensor, control_targets: Tensor | None,
                             future_control_targets: Tensor | None = None) -> Tensor:
        if not self.config.use_prefix_future:
            return state.mean() * 0.0
        pred = self.prefix_future_head(state)
        if future_control_targets is None and control_targets is not None:
            future_control_targets = control_targets[:, -pred.shape[1]:, :]
        if future_control_targets is None:
            return pred.abs().mean()
        return masked_l2_loss(pred, future_control_targets.to(device=pred.device, dtype=pred.dtype), self.config.invalid_control_value)

    def forward(self, frames: Tensor | None = None, control_targets: Tensor | None = None,
                intervention: InterventionSpec | None = None, batch: ACPRFlowBatch | None = None,
                masked_ids: Tensor | None = None, raw_actions: list[str] | None = None,
                raw_justifications: list[str] | None = None,
                reason_semantic_target: Tensor | None = None,
                future_control_targets: Tensor | None = None,
                precomputed_grids: dict[str, Tensor] | None = None) -> ACPRFlowTrainOutput:
        frames, control_targets, masked_ids, raw_actions, raw_justifications = self._resolve_inputs(
            frames, batch, control_targets, masked_ids, raw_actions, raw_justifications
        )
        bundle = self.build_bundle(frames, intervention=intervention, precomputed_grids=precomputed_grids)
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
        action_logits = enhanced_masked_logits + 0.01 * self.reason_to_action(enhanced_hidden)
        explanation_logits = enhanced_masked_logits + 0.01 * self.reason_to_explanation(enhanced_hidden)
        action_loss = self._masked_token_loss(action_logits, masked_ids)
        explanation_loss = self._masked_token_loss(explanation_logits, masked_ids)
        if control_targets is None:
            control_loss = control_final.sum() * 0.0
        else:
            control_targets = control_targets.to(device=control_final.device, dtype=control_final.dtype)
            control_loss = masked_l2_loss(control_final, control_targets, self.config.invalid_control_value)
        text_targets = self.text_targets.build(raw_actions, raw_justifications, device=frames.device)
        predicate_pu = partial_label_bce(
            bundle.predicate_logits_temporal.mean(dim=1),
            text_targets["predicate_positive"],
            text_targets["predicate_contradiction"],
            text_targets["predicate_reliability"],
        )
        flow_pu = partial_label_bce(
            bundle.flow_logits,
            text_targets["flow_positive"],
            text_targets["flow_contradiction"],
            text_targets["flow_reliability"],
        )
        reason_semantic = self._reason_semantic_loss(bundle.global_reason_state, reason_semantic_target)
        future_control = self._future_control_loss(bundle.global_reason_state, control_targets, future_control_targets)
        memory_diversity = memory_diversity_loss(bundle.reason_memory)
        auxiliary_loss = (
            0.001 * memory_diversity
            + 0.05 * predicate_pu
            + 0.03 * flow_pu
            + 0.05 * reason_semantic
            + 0.02 * future_control
        )
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
                "reason_semantic": reason_semantic,
                "future_control": future_control,
                "memory_diversity": memory_diversity,
                "total": total,
            },
            bundle=bundle,
        )
