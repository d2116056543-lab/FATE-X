from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import DynFlowConfig
from .covariate_homogenizer import PredicateCovariateHomogenizer
from .decision_ledger import DecisionLedgerHead
from .dynamic_predicate_field import ACPRDynamicPredicateField
from .global_decision_stream import GlobalDecisionStream
from .mesoscopic_lane_flow import MesoscopicLaneFlow
from .multiscale_pattern_router import MultiScalePatternRouter
from .nnpu_calalign import OnlineCalAlign, PULabels, nnpu_risk
from .predicate_ontology import EXACT_32_PREDICATES
from .predicate_transfer import PredicateQueryInitializer
from .response_lag import ResponseLagAligner
from .signal_codec import BDDSignalCodec
from .text_decoder import DynFlowTextDecoder
from .traffic_state_reasoner import TrafficStateReasoner
from .types import ACPRDynFlowOutput, DynFlowBatch
from .video_backbone import ACPRDynFlowVideoBackbone


class ACPRDynFlowModel(nn.Module):
    def __init__(self, cfg: DynFlowConfig | dict[str, Any] | None = None, codec: BDDSignalCodec | None = None):
        super().__init__()
        raw = cfg.raw if isinstance(cfg, DynFlowConfig) else (cfg or {})
        model_cfg = raw.get("model", {})
        state_dim = int(model_cfg.get("state_dim", 256))
        decision_dim = int(model_cfg.get("decision_dim", 512))
        text_dim = int(model_cfg.get("text_hidden_dim", 768))
        paths = raw.get("paths", {})
        self.codec = codec or BDDSignalCodec()
        self.backbone = ACPRDynFlowVideoBackbone(state_dim=state_dim, text_dim=text_dim, checkpoint_path=paths.get("video_swin_kinetics_checkpoint"))
        self.query_init = PredicateQueryInitializer(dim=state_dim, checkpoint_path=paths.get("oia_acpr_checkpoint"))
        self.predicates = ACPRDynamicPredicateField(dim=state_dim)
        self.homogenizer = PredicateCovariateHomogenizer(out_dim=state_dim)
        self.pattern_router = MultiScalePatternRouter(dim=state_dim)
        self.lane_flow = MesoscopicLaneFlow(dim=state_dim)
        self.reasoner = TrafficStateReasoner(dim=state_dim)
        self.lag = ResponseLagAligner(dim=state_dim)
        self.global_decision = GlobalDecisionStream(dim=state_dim, decision_dim=decision_dim)
        self.ledger_head = DecisionLedgerHead(dim=state_dim, signal_names=self.codec.signal_names)
        self.text_decoder = DynFlowTextDecoder(text_dim=text_dim, factor_dim=state_dim, bert_dir=paths.get("bert_dir"))
        self.calalign = OnlineCalAlign(num_predicates=len(EXACT_32_PREDICATES))

    def forward(self, batch: DynFlowBatch, intervention: str | None = None) -> ACPRDynFlowOutput:
        frames = batch.frames
        if intervention == "temporal_reverse":
            frames = torch.flip(frames, dims=[1])
        elif intervention == "temporal_shuffle":
            frames = frames[:, torch.randperm(frames.shape[1], device=frames.device)]
        elif intervention == "last_frame_only":
            frames = frames[:, -1:].expand_as(frames)
        bb = self.backbone(frames)
        q, q_report = self.query_init()
        pred = self.predicates(bb.local_grid, bb.coarse_grid, q)
        if intervention in {"predicate_off", "evidence_tube_off"}:
            pred.probabilities = pred.probabilities * 0.0
            pred.tokens = pred.tokens * 0.0
        cov = self.homogenizer(pred)
        cov = self.pattern_router(cov)
        lane = self.lane_flow(pred, cov)
        flow = self.reasoner(pred, cov, lane)
        if intervention in {"all_flow_off", "regime_off", "phase_off", "source_off", "top_factor_off"}:
            flow.factor_tokens = flow.factor_tokens * 0.0
            flow.factor_probs = flow.factor_probs * 0.0
        flow = self.lag(flow, target_steps=32)
        if intervention == "lag_disabled":
            flow.lag_aligned_tokens = flow.factor_tokens.repeat_interleave((32 + flow.factor_tokens.shape[1] - 1) // flow.factor_tokens.shape[1], dim=1)[:, :32]
        global_norm, global_state = self.global_decision(bb.global_sequence)
        ledger = self.ledger_head(global_norm, flow, self.codec)
        text = self.text_decoder(batch.input_ids, batch.masked_ids, flow, ledger)
        losses: dict[str, Tensor] = {}
        if batch.control_target is not None:
            target_norm = self.codec.encode(batch.control_target.to(ledger.final_prediction_normalized.device).float())
            mask = target_norm.ne(float(self.codec.invalid_value))
            control = F.smooth_l1_loss(ledger.final_prediction_normalized[mask], target_norm[mask]) if bool(mask.any()) else ledger.final_prediction_normalized.sum() * 0
            losses["final_speed_normalized"] = control
            losses["final_course_normalized"] = control
            losses["global_speed_normalized"] = F.smooth_l1_loss(ledger.global_prediction_normalized[mask], target_norm[mask]) if bool(mask.any()) else control * 0
            losses["global_course_normalized"] = losses["global_speed_normalized"]
        else:
            zero = ledger.final_prediction_normalized.sum() * 0
            losses.update({k: zero for k in ("final_speed_normalized", "final_course_normalized", "global_speed_normalized", "global_course_normalized")})
        labels = PULabels(
            positive=torch.zeros(frames.shape[0], len(EXACT_32_PREDICATES), device=frames.device),
            reliable_negative=torch.zeros(frames.shape[0], len(EXACT_32_PREDICATES), device=frames.device),
            unlabeled=torch.ones(frames.shape[0], len(EXACT_32_PREDICATES), device=frames.device),
        )
        losses["predicate_nnpu"] = nnpu_risk(pred.logits.mean(1), labels, self.calalign.prior)
        losses["predicate_query_anchor"] = q.pow(2).mean() * 0.01
        losses["pattern_semantic"] = cov.pattern_probs.mean() * 0.0 + cov.pattern_logits.pow(2).mean() * 1e-4
        losses["traffic_grammar"] = flow.factor_probs.mean() * 1e-4
        losses["contribution_alignment"] = (text.action_to_factor_attention.mean(1) - ledger.speed_factor_attention).abs().mean()
        losses["temporal_consistency"] = pred.relative_centroid_motion.abs().mean()
        losses["contribution_sparsity"] = ledger.factor_contributions_normalized.abs().mean()
        losses["contribution_smoothness"] = (ledger.factor_contributions_normalized[:, 1:] - ledger.factor_contributions_normalized[:, :-1]).abs().mean()
        losses["action_text"] = text.action_loss
        losses["explanation_text"] = text.explanation_loss
        losses["flow_residual_speed"] = ledger.factor_contributions_normalized[..., 1].abs().mean()
        losses["flow_residual_course"] = ledger.factor_contributions_normalized[..., 0].abs().mean()
        losses["control_first_difference"] = (ledger.final_prediction_normalized[:, 1:] - ledger.final_prediction_normalized[:, :-1]).abs().mean()
        total = sum(losses.values())
        return ACPRDynFlowOutput(
            total_loss=total,
            loss_components=losses,
            backbone=bb,
            predicates=pred,
            covariates=cov,
            flow=flow,
            ledger=ledger,
            text=text,
            diagnostics={"query_transfer": q_report, "intervention": intervention},
        )

