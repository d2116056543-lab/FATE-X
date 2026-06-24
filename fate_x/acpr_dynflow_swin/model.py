from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .decision_ledger import ExactDecisionLedgerHead
from .dynamic_predicate_field import DynamicPredicateFieldBuilder
from .interventions import InterventionSpec, temporal_reverse_frames, zero_factor
from .pattern_lag_traffic_reasoner import PatternLagTrafficReasoner
from .nnpu_calalign import OnlineCalAlign, PredicateRuleLabeler, nnpu_loss
from .predicate_transfer import PredicateQueryTransfer, load_bert_name_features
from .query_motion_transformer import QueryMotionTransformer
from .semantic_token_consolidator import SemanticTokenConsolidator
from .text_decoder import DynFlowSwinTextDecoder, build_generic_adapt_captioner
from .types import ACPRDynFlowSwinOutput, DynFlowSwinBatch
from .video_swin_backbone import ACPRDynFlowSwinBackbone
from fate_x.losses.acpr_dynflow_swin_losses import (
    benefit_gate_loss,
    contribution_alignment_js,
    group_sparsity,
    non_degradation_hinge,
    pattern_semantic_loss,
    pattern_targets_from_predicates,
    residual_target_loss,
    target_delta_loss,
    temporal_smoothness,
    traffic_state_semantic_loss,
    traffic_targets_from_predicates,
    weighted_loss_total,
)


def _dim(cfg: dict[str, Any], key: str, default: int) -> int:
    return int(cfg.get("model", {}).get("dimensions", {}).get(key, default))


class ACPRDynFlowSwinModel(nn.Module):
    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = cfg or {}
        predicate_dim = _dim(self.cfg, "predicate", 256)
        text_dim = _dim(self.cfg, "text", 768)
        motion_dim = _dim(self.cfg, "motion", 768)
        self.backbone = ACPRDynFlowSwinBackbone(self.cfg)
        self.predicate_input_proj = nn.LazyLinear(predicate_dim)
        self.final_input_proj = nn.LazyLinear(predicate_dim)
        self.predicates = DynamicPredicateFieldBuilder(dim=predicate_dim)
        paths = self.cfg.get("paths", {})
        bert_dir = paths.get("bert_dir")
        name_features = load_bert_name_features(bert_dir) if bert_dir and torch.jit.is_scripting() is False else None
        self.predicate_transfer = PredicateQueryTransfer(
            dim=predicate_dim,
            gate_init=float(self.cfg.get("model", {}).get("oia_transfer", {}).get("transfer_gate_init", 0.25)),
            name_features=name_features,
        )
        oia_checkpoint = paths.get("oia_acpr_checkpoint")
        if oia_checkpoint and str(oia_checkpoint).upper() not in {"", "REQUIRED"}:
            self.predicate_transfer.load_oia_query(
                oia_checkpoint, self.cfg.get("model", {}).get("oia_transfer", {}).get(
                    "tensor_key", "model.predicate_head.predicate_queries"
                )
            )
        rules_path = self.cfg.get("model", {}).get("nnpu", {}).get(
            "rules_yaml", paths.get("nnpu_rules", "configs/acpr_dynflow_swin_text_rules.yaml")
        )
        self.predicate_labeler = PredicateRuleLabeler.from_yaml(rules_path)
        self.calalign = OnlineCalAlign(len(self.predicate_labeler.rules))
        self.consolidator = SemanticTokenConsolidator(input_dim=predicate_dim, output_dim=motion_dim)
        self.traffic = PatternLagTrafficReasoner(predicate_dim=predicate_dim, factor_dim=motion_dim)
        self.global_to_motion = nn.LazyLinear(motion_dim)
        self.motion = QueryMotionTransformer(input_dim=motion_dim, hidden_dim=motion_dim)
        self.ledger = ExactDecisionLedgerHead(dim=motion_dim, factor_count=13, signal_names=("course", "speed"))
        self.text_context = nn.Linear(motion_dim, text_dim)
        self.contribution_to_text = nn.Linear(2, text_dim)
        captioner = tokenizer = None
        if paths.get("bert_dir") and self.cfg.get("model", {}).get("text", {}).get(
            "architecture"
        ) == "repository_bert_for_image_captioning":
            captioner, tokenizer = build_generic_adapt_captioner(paths["bert_dir"], img_feature_dim=text_dim)
        self.text = DynFlowSwinTextDecoder(
            hidden_dim=text_dim,
            factor_dim=motion_dim,
            bert_captioner=captioner,
            tokenizer=tokenizer,
        )

    def forward(
        self,
        batch: DynFlowSwinBatch,
        intervention: InterventionSpec | None = None,
        generate_text: bool = False,
    ) -> ACPRDynFlowSwinOutput:
        frames = batch.frames
        trace = {
            "kind": "none",
            "earliest_layer": None,
            "rerun_layers": [],
        }
        if intervention is not None:
            trace["kind"] = intervention.kind
            if intervention.kind == "temporal_reverse":
                frames = temporal_reverse_frames(frames)
                trace["earliest_layer"] = "frames"
                trace["rerun_layers"] = [
                    "backbone", "predicates", "semantic_consolidation", "traffic", "motion", "ledger", "text"
                ]
            elif intervention.kind == "factor_off":
                trace["earliest_layer"] = "traffic"
                trace["rerun_layers"] = ["ledger", "text"]
            else:
                raise ValueError(f"unsupported intervention {intervention.kind}")
        bb = self.backbone(frames)
        predicate_grid = self.predicate_input_proj(bb.predicate_grid)
        native_final_steps = int(bb.final_grid.shape[1])
        final_tokens = self.final_input_proj(bb.dense_final_tokens).reshape(batch.frames.shape[0], native_final_steps, 49, -1)
        base_queries, transfer_report = self.predicate_transfer()
        pred = self.predicates(predicate_grid, base_queries=base_queries)
        sem = self.consolidator(final_tokens)
        traffic = self.traffic(pred.tokens, pred.evidence_maps, pred.corridor_mass, target_steps=32)
        if intervention is not None and intervention.kind == "factor_off":
            if intervention.factor_index is None:
                raise ValueError("factor_off requires factor_index")
            traffic = zero_factor(traffic, intervention.factor_index)
        motion = self.motion(self.global_to_motion(bb.temporal_global), sem.tokens)
        ledger = self.ledger(motion.global_prediction_normalized, traffic.lag_aligned_tokens)
        text_hidden = self.text_context(motion.query_hidden[:, : batch.input_ids.shape[1]])
        if text_hidden.shape[1] < batch.input_ids.shape[1]:
            repeat = batch.input_ids.shape[1] - text_hidden.shape[1]
            text_hidden = torch.cat([text_hidden, text_hidden[:, -1:].expand(-1, repeat, -1)], dim=1)
        semantic_visual = sem.tokens.reshape(sem.tokens.shape[0], -1, sem.tokens.shape[-1])
        factor_visual = traffic.lag_aligned_tokens.mean(dim=1)
        contribution_visual = self.contribution_to_text(
            ledger.gated_factor_contributions_normalized.mean(dim=1)
        )
        visual_tokens = torch.cat([semantic_visual, factor_visual, contribution_visual], dim=1)
        text = self.text(
            batch.input_ids,
            batch.masked_pos,
            batch.masked_ids,
            text_hidden,
            traffic.lag_aligned_tokens,
            token_type_ids=batch.token_type_ids,
            attention_mask=batch.attention_mask,
            visual_tokens=visual_tokens,
        )
        if generate_text:
            action_ids, explanation_ids = self.text.generate_text(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                masked_pos=batch.masked_pos,
                token_type_ids=batch.token_type_ids,
                img_feats=visual_tokens,
            )
            text.generated_action = self.text.decode_token_ids(action_ids)
            text.generated_explanation = self.text.decode_token_ids(explanation_ids)
        raw_losses: dict[str, Tensor] = {
            "action_text": text.action_loss,
            "explanation_text": text.explanation_loss,
        }
        if batch.control_target is not None:
            target = batch.control_target.to(ledger.final_prediction_normalized.device).to(ledger.final_prediction_normalized.dtype)
            raw_losses["final_course_normalized"] = F.smooth_l1_loss(ledger.final_prediction_normalized[..., 0], target[..., 0])
            raw_losses["final_speed_normalized"] = F.smooth_l1_loss(ledger.final_prediction_normalized[..., 1], target[..., 1])
            raw_losses["global_course_normalized"] = F.smooth_l1_loss(ledger.global_prediction_normalized[..., 0], target[..., 0])
            raw_losses["global_speed_normalized"] = F.smooth_l1_loss(ledger.global_prediction_normalized[..., 1], target[..., 1])
            residual_target = (target - ledger.global_prediction_normalized).detach()
            raw_losses["flow_residual_course"] = residual_target_loss(
                ledger.gated_factor_contributions_normalized[..., 0:1],
                residual_target[..., 0:1],
            )
            raw_losses["flow_residual_speed"] = residual_target_loss(
                ledger.gated_factor_contributions_normalized[..., 1:2],
                residual_target[..., 1:2],
            )
            full_error = (ledger.final_prediction_normalized - target).abs()
            global_error = (ledger.global_prediction_normalized - target).abs()
            without_factor = (
                ledger.final_prediction_normalized.unsqueeze(2)
                - ledger.gated_factor_contributions_normalized
            )
            without_error = (without_factor - target.unsqueeze(2)).abs()
            margin = float(
                self.cfg.get("model", {}).get("ledger", {}).get(
                    "non_degradation_margin_normalized", 0.01
                )
            )
            benefit_target = without_error.gt(full_error.unsqueeze(2) + margin).to(
                ledger.benefit_gate.dtype
            )
            ledger.benefit_target = benefit_target
            raw_losses["benefit_gate"] = benefit_gate_loss(ledger.benefit_gate, benefit_target)
            raw_losses["non_degradation"] = non_degradation_hinge(
                full_error, global_error, margin=margin
            )
            raw_losses["control_delta_match"] = target_delta_loss(
                ledger.final_prediction_normalized, target
            )
        else:
            zero = ledger.final_prediction_normalized.sum() * 0.0
            raw_losses.update(
                {
                    key: zero
                    for key in (
                        "final_course_normalized",
                        "final_speed_normalized",
                        "global_course_normalized",
                        "global_speed_normalized",
                        "flow_residual_course",
                        "flow_residual_speed",
                        "benefit_gate",
                        "non_degradation",
                        "control_delta_match",
                    )
                }
            )
        texts = [
            f"{action} {justification}".strip()
            for action, justification in zip(batch.raw_actions, batch.raw_justifications)
        ]
        pu_labels = self.predicate_labeler.label(texts, device=pred.logits.device)
        predicate_logits = pred.logits.mean(dim=1)
        raw_losses["predicate_nnpu"] = nnpu_loss(predicate_logits, pu_labels, self.calalign.prior)
        self.calalign.update(predicate_logits, pu_labels, training=self.training)
        raw_losses["predicate_transfer_anchor"] = self.predicate_transfer.domain_residual.pow(2).mean()
        pattern_target = pattern_targets_from_predicates(pred.probabilities.detach())
        if pattern_target.shape[1] != traffic.pattern_logits.shape[1]:
            pattern_target = F.interpolate(
                pattern_target.float().unsqueeze(1),
                size=traffic.pattern_logits.shape[1],
                mode="nearest",
            ).squeeze(1).long()
        raw_losses["pattern_semantic"] = pattern_semantic_loss(
            traffic.pattern_logits, pattern_target
        )
        traffic_target = traffic_targets_from_predicates(
            pred.probabilities.detach(),
            pattern_targets_from_predicates(pred.probabilities.detach()),
        )
        if traffic_target.shape[1] != traffic.factor_logits.shape[1]:
            traffic_target = F.interpolate(
                traffic_target.transpose(1, 2),
                size=traffic.factor_logits.shape[1],
                mode="nearest",
            ).transpose(1, 2)
        raw_losses["traffic_state_semantic"] = traffic_state_semantic_loss(
            traffic.factor_logits, traffic_target
        )
        contribution_importance = ledger.gated_factor_contributions_normalized.abs().sum(dim=-1)
        raw_losses["contribution_alignment"] = contribution_alignment_js(
            text.explanation_to_factor_attention,
            contribution_importance.mean(dim=1),
        )
        raw_losses["contribution_group_sparsity"] = group_sparsity(
            ledger.gated_factor_contributions_normalized
        )
        raw_losses["contribution_smoothness"] = temporal_smoothness(
            ledger.gated_factor_contributions_normalized
        )
        loss_weights = self.cfg.get("loss") or {name: 1.0 for name in raw_losses}
        total, weighted_losses = weighted_loss_total(raw_losses, loss_weights)
        loss_components = {
            **weighted_losses,
            **{f"raw/{name}": value for name, value in raw_losses.items()},
        }
        return ACPRDynFlowSwinOutput(
            total_loss=total,
            loss_components=loss_components,
            backbone=bb,
            predicates=pred,
            semantic_tokens=sem,
            traffic=traffic,
            motion=motion,
            ledger=ledger,
            text=text,
            diagnostics={
                "formal_namespace": "fate_x.acpr_dynflow_swin",
                "oia_transfer": transfer_report,
                "nnpu_counts": {
                    "positive": int(pu_labels.positive.sum().detach().cpu()),
                    "reliable_negative": int(pu_labels.reliable_negative.sum().detach().cpu()),
                    "unlabeled": int(pu_labels.unlabeled.sum().detach().cpu()),
                },
                "calalign_update_count": int(self.calalign.update_count.detach().cpu()),
                "intervention": trace,
                "loss_weights": {
                    name: float(weight)
                    for name, weight in loss_weights.items()
                },
            },
        )
