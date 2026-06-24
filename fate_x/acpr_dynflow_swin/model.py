from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .decision_ledger import ExactDecisionLedgerHead
from .dynamic_predicate_field import DynamicPredicateFieldBuilder
from .pattern_lag_traffic_reasoner import PatternLagTrafficReasoner
from .query_motion_transformer import QueryMotionTransformer
from .semantic_token_consolidator import SemanticTokenConsolidator
from .text_decoder import DynFlowSwinTextDecoder
from .types import ACPRDynFlowSwinOutput, DynFlowSwinBatch
from .video_swin_backbone import ACPRDynFlowSwinBackbone


def _dim(cfg: dict[str, Any], key: str, default: int) -> int:
    return int(cfg.get("model", {}).get("dimensions", {}).get(key, default))


class ACPRDynFlowSwinModel(nn.Module):
    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = cfg or {}
        predicate_dim = _dim(self.cfg, "predicate", 64)
        text_dim = _dim(self.cfg, "text", 768)
        motion_dim = _dim(self.cfg, "motion", text_dim)
        self.backbone = ACPRDynFlowSwinBackbone(out_dim=predicate_dim)
        self.predicates = DynamicPredicateFieldBuilder(dim=predicate_dim)
        self.consolidator = SemanticTokenConsolidator(input_dim=predicate_dim, output_dim=motion_dim)
        self.traffic = PatternLagTrafficReasoner(predicate_dim=predicate_dim, factor_dim=motion_dim)
        self.global_to_motion = nn.Linear(predicate_dim, motion_dim)
        self.motion = QueryMotionTransformer(input_dim=motion_dim, hidden_dim=motion_dim)
        self.ledger = ExactDecisionLedgerHead(dim=motion_dim, factor_count=13, signal_names=("course", "speed"))
        self.text_context = nn.Linear(motion_dim, text_dim)
        self.text = DynFlowSwinTextDecoder(hidden_dim=text_dim, factor_dim=motion_dim)

    def forward(self, batch: DynFlowSwinBatch) -> ACPRDynFlowSwinOutput:
        bb = self.backbone(batch.frames)
        pred = self.predicates(bb.predicate_grid)
        sem = self.consolidator(bb.dense_final_tokens.reshape(batch.frames.shape[0], 32, 49, -1)[:, ::2])
        traffic = self.traffic(pred.tokens, pred.evidence_maps, pred.corridor_mass, target_steps=32)
        motion = self.motion(self.global_to_motion(bb.temporal_global), sem.tokens)
        ledger = self.ledger(motion.global_prediction_normalized, traffic.lag_aligned_tokens)
        text_hidden = self.text_context(motion.query_hidden[:, : batch.input_ids.shape[1]])
        if text_hidden.shape[1] < batch.input_ids.shape[1]:
            repeat = batch.input_ids.shape[1] - text_hidden.shape[1]
            text_hidden = torch.cat([text_hidden, text_hidden[:, -1:].expand(-1, repeat, -1)], dim=1)
        text = self.text(batch.input_ids, batch.masked_pos, batch.masked_ids, text_hidden, traffic.lag_aligned_tokens)
        losses: dict[str, Tensor] = {
            "action_text": text.action_loss,
            "explanation_text": text.explanation_loss,
        }
        if batch.control_target is not None:
            target = batch.control_target.to(ledger.final_prediction_normalized.device).to(ledger.final_prediction_normalized.dtype)
            losses["final_course_normalized"] = F.smooth_l1_loss(ledger.final_prediction_normalized[..., 0], target[..., 0])
            losses["final_speed_normalized"] = F.smooth_l1_loss(ledger.final_prediction_normalized[..., 1], target[..., 1])
            losses["global_course_normalized"] = F.smooth_l1_loss(ledger.global_prediction_normalized[..., 0], target[..., 0])
            losses["global_speed_normalized"] = F.smooth_l1_loss(ledger.global_prediction_normalized[..., 1], target[..., 1])
        else:
            zero = ledger.final_prediction_normalized.sum() * 0.0
            losses.update({k: zero for k in ("final_course_normalized", "final_speed_normalized", "global_course_normalized", "global_speed_normalized")})
        losses["predicate_nnpu"] = pred.logits.mean().abs() * 0.0
        losses["contribution_alignment"] = (
            text.explanation_to_factor_attention - ledger.speed_factor_attention.mean(dim=1)
        ).abs().mean()
        total = sum(losses.values())
        return ACPRDynFlowSwinOutput(
            total_loss=total,
            loss_components=losses,
            backbone=bb,
            predicates=pred,
            semantic_tokens=sem,
            traffic=traffic,
            motion=motion,
            ledger=ledger,
            text=text,
            diagnostics={"formal_namespace": "fate_x.acpr_dynflow_swin"},
        )
