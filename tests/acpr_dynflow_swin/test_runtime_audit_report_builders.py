from __future__ import annotations

import torch

from fate_x.acpr_dynflow_swin.types import (
    ACPRDynFlowSwinOutput,
    DynamicPredicateField,
    DynFlowSwinTextOutput,
    ExactDecisionLedger,
    MotionTransformerOutput,
    SemanticTokenConsolidation,
    SwinBackboneOutput,
    TrafficStateOutput,
)
from fate_x.engine.audit_acpr_dynflow_swin_runtime import build_runtime_gate_reports


def _output() -> ACPRDynFlowSwinOutput:
    b, tm, th, p, f, d = 1, 4, 4, 32, 13, 8
    final = torch.randn(b, 32, 2)
    global_pred = torch.randn(b, 32, 2)
    raw = torch.randn(b, 32, f, 2) * 0.01
    gate = torch.sigmoid(torch.randn(b, 32, 2))
    gated = raw * gate.unsqueeze(2)
    return ACPRDynFlowSwinOutput(
        total_loss=torch.tensor(3.0, requires_grad=True),
        loss_components={
            "final_speed_normalized": torch.tensor(0.2),
            "final_course_normalized": torch.tensor(0.3),
            "action_text": torch.tensor(0.4),
            "explanation_text": torch.tensor(0.5),
            "predicate_nnpu": torch.tensor(0.1),
            "raw/predicate_nnpu": torch.tensor(1.25),
        },
        backbone=SwinBackboneOutput(
            predicate_grid=torch.randn(b, tm, 2, 2, d),
            final_grid=torch.randn(b, th, 7, 7, 16),
            temporal_global=torch.randn(b, th, 16),
            dense_final_tokens=torch.randn(b, th * 49, 16),
            forward_count=1,
        ),
        predicates=DynamicPredicateField(
            names=tuple(f"p{i}" for i in range(p)),
            query_states=torch.randn(b, tm, p, d),
            logits=torch.randn(b, tm, p),
            probabilities=torch.rand(b, tm, p),
            tokens=torch.randn(b, tm, p, d),
            evidence_maps=torch.softmax(torch.randn(b, tm, p, 2, 2).flatten(-2), dim=-1).view(b, tm, p, 2, 2),
            confidence=torch.rand(b, tm, p),
            centroid=torch.rand(b, tm, p, 2),
            relative_motion=torch.randn(b, tm - 1, p, 2),
            corridor_mass=torch.rand(b, tm, p, 3),
            transfer_gate=torch.full((p,), 0.25),
        ),
        semantic_tokens=SemanticTokenConsolidation(
            slot_names=("global", "long", "left", "right", "residual"),
            assignment=torch.softmax(torch.randn(b, th, 49, 5), dim=-1),
            token_mass=torch.rand(b, th, 5),
            tokens=torch.randn(b, th, 5, d),
            source_provenance=torch.softmax(torch.randn(b, th, 49, 5), dim=-1),
            conservation_error=torch.tensor(0.0),
        ),
        traffic=TrafficStateOutput(
            factor_names=tuple(f"factor{i}" for i in range(f)),
            factor_tokens_native=torch.randn(b, tm, f, d),
            factor_logits=torch.randn(b, tm, f),
            factor_probs=torch.rand(b, tm, f),
            lateral_bias=torch.randn(b, tm, 1),
            pattern_logits=torch.randn(b, tm, 4),
            pattern_probs=torch.softmax(torch.randn(b, tm, 4), dim=-1),
            factor_to_predicate=torch.softmax(torch.randn(b, tm, f, p), dim=-1),
            factor_to_corridor=torch.softmax(torch.randn(b, tm, f, 3), dim=-1),
            evidence_maps=torch.softmax(torch.randn(b, tm, f, 2, 2).flatten(-2), dim=-1).view(b, tm, f, 2, 2),
            lag_weights=torch.softmax(torch.randn(b, 32, f, 4), dim=-1),
            lag_aligned_tokens=torch.randn(b, 32, f, d),
            lineage=[{"factor": "factor0", "predicate_support": ["p0"]}],
        ),
        motion=MotionTransformerOutput(
            query_hidden=torch.randn(b, 32, 768),
            global_prediction_normalized=global_pred,
            source_attention=torch.rand(b, 32, 80),
        ),
        ledger=ExactDecisionLedger(
            signal_names=("course", "speed"),
            global_prediction_normalized=global_pred,
            raw_factor_contributions_normalized=raw,
            benefit_gate=gate,
            gated_factor_contributions_normalized=gated,
            final_prediction_normalized=global_pred + gated.sum(dim=2),
            global_prediction_raw=global_pred,
            gated_factor_contributions_raw=gated,
            final_prediction_raw=global_pred + gated.sum(dim=2),
            speed_factor_attention=torch.softmax(torch.randn(b, 32, f), dim=-1),
            course_factor_attention=torch.softmax(torch.randn(b, 32, f), dim=-1),
            benefit_target=torch.rand(b, 32, 2),
        ),
        text=DynFlowSwinTextOutput(
            total_mlm_loss=torch.tensor(0.9),
            action_loss=torch.tensor(0.4),
            explanation_loss=torch.tensor(0.5),
            action_logits=torch.randn(b, 15, 30522),
            explanation_logits=torch.randn(b, 15, 30522),
            action_to_factor_attention=torch.softmax(torch.randn(b, 15, f), dim=-1),
            explanation_to_factor_attention=torch.softmax(torch.randn(b, 15, f), dim=-1),
            generated_action=["slow down"],
            generated_explanation=["because traffic is slowing"],
        ),
        diagnostics={
            "nnpu_counts": {"positive": 3, "reliable_negative": 4, "unlabeled": 121},
            "calalign_update_count": 1,
            "loss_weights": {"predicate_nnpu": 0.08},
        },
    )


def test_runtime_gate_reports_are_tensor_linked_and_non_placeholder():
    reports = build_runtime_gate_reports(_output(), gradient_abs_sum={"predicates": 1.0, "traffic": 2.0, "text": 3.0})
    required = {
        "tensor_contracts.json",
        "video_swin_backbone_audit.json",
        "dynamic_predicate_audit.json",
        "semantic_consolidation_audit.json",
        "pattern_traffic_audit.json",
        "response_lag_audit.json",
        "query_motion_transformer_audit.json",
        "decision_ledger_audit.json",
        "text_decoder_audit.json",
        "gradient_direction_audit.json",
        "loss_audit.json",
        "gate_gradient_chain.json",
        "gate_identity_checks.json",
        "gate_temporal_lag.json",
    }
    assert required <= set(reports)
    assert all(payload["passed"] is True for payload in reports.values())
    assert all("requires real" not in str(payload).lower() for payload in reports.values())
    assert reports["decision_ledger_audit.json"]["normalized_reconstruction_max_error"] < 1e-5
    assert reports["text_decoder_audit.json"]["generated_action"] == ["slow down"]
