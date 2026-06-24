from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from fate_x.acpr_dynflow_swin.config import load_config
from fate_x.acpr_dynflow_swin.interventions import InterventionSpec, run_intervention
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.acpr_dynflow_swin.nnpu_calalign import PredicateRuleLabeler
from fate_x.acpr_dynflow_swin.predicate_transfer import PredicateQueryTransfer, load_bert_name_features
from fate_x.acpr_dynflow_swin.types import ACPRDynFlowSwinOutput
from fate_x.engine.acpr_dynflow_swin_data import build_dynflow_swin_dataloader
from fate_x.engine.eval_acpr_dynflow_swin import move_batch_to_device
from fate_x.explain.acpr_dynflow_swin_atlas import build_atlas
from fate_x.explain.acpr_dynflow_swin_renderer import render_case_canvas, tensor_case_from_output


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_gitdir_to_wsl(path_text: str) -> str:
    text = path_text.strip().replace("\\", "/")
    if len(text) > 2 and text[1:3] == ":/":
        return f"/mnt/{text[0].lower()}/{text[3:]}"
    return text


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        git_file = Path(".git")
        if not git_file.is_file():
            raise
        text = git_file.read_text(encoding="utf-8").strip()
        if not text.lower().startswith("gitdir:"):
            raise
        env = os.environ.copy()
        env["GIT_DIR"] = _windows_gitdir_to_wsl(text.split(":", 1)[1])
        env["GIT_WORK_TREE"] = str(Path.cwd())
        return subprocess.check_output(["git", *args], text=True, env=env, stderr=subprocess.DEVNULL).strip()


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().cpu())


def _shape(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def _max_abs(tensor: torch.Tensor) -> float:
    return float(tensor.detach().abs().max().cpu())


def _mean_abs(tensor: torch.Tensor) -> float:
    return float(tensor.detach().abs().mean().cpu())


def _passed(payload: dict[str, Any], condition: bool, reason: str = "") -> dict[str, Any]:
    payload["passed"] = bool(condition)
    payload["status"] = "pass" if condition else "blocked"
    if not condition and reason:
        payload["reason"] = reason
    return payload


def build_runtime_gate_reports(
    output: ACPRDynFlowSwinOutput,
    gradient_abs_sum: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Build tensor-linked formal gate reports from one real model output.

    This function intentionally contains no file-system shortcuts. Every pass
    decision is derived from tensors emitted by the forward/backward path so the
    preflight cannot replace the dynamic gates with placeholder JSON.
    """

    reports: dict[str, dict[str, Any]] = {}
    dense_assignment_sum = output.semantic_tokens.assignment.sum(dim=-1)
    semantic_mass_error = _max_abs(output.semantic_tokens.conservation_error.float())
    norm_recon = output.ledger.global_prediction_normalized + output.ledger.gated_factor_contributions_normalized.sum(dim=2)
    raw_recon = output.ledger.global_prediction_raw + output.ledger.gated_factor_contributions_raw.sum(dim=2)
    norm_recon_error = _max_abs(norm_recon - output.ledger.final_prediction_normalized)
    raw_recon_error = _max_abs(raw_recon - output.ledger.final_prediction_raw)
    lag_sum_error = _max_abs(output.traffic.lag_weights.sum(dim=-1) - 1.0)
    evidence_sum_error = _max_abs(output.predicates.evidence_maps.flatten(-2).sum(dim=-1) - 1.0)
    corridor_variance = float(output.traffic.factor_to_corridor.detach().float().var(dim=-1).mean().cpu())
    pattern_variance = float(output.traffic.pattern_probs.detach().float().var(dim=-1).mean().cpu())
    predicate_temporal_delta = _mean_abs(output.predicates.probabilities[:, 1:] - output.predicates.probabilities[:, :-1])
    lag_temporal_delta = _mean_abs(output.traffic.lag_aligned_tokens[:, 1:] - output.traffic.lag_aligned_tokens[:, :-1])
    generated_action = output.text.generated_action or []
    generated_explanation = output.text.generated_explanation or []
    action_explanation_distinct = output.text.action_loss is not output.text.explanation_loss
    loss_components = {
        name: float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)
        for name, value in output.loss_components.items()
    }
    nonzero_loss_names = [
        name for name, value in loss_components.items()
        if not name.startswith("raw/") and abs(value) > 0
    ]
    grad_nonzero = {name: value for name, value in gradient_abs_sum.items() if value > 0}

    reports["tensor_contracts.json"] = _passed({
        "backbone": {
            "predicate_grid": _shape(output.backbone.predicate_grid),
            "final_grid": _shape(output.backbone.final_grid),
            "temporal_global": _shape(output.backbone.temporal_global),
            "dense_final_tokens": _shape(output.backbone.dense_final_tokens),
        },
        "predicates": {
            "names": len(output.predicates.names),
            "query_states": _shape(output.predicates.query_states),
            "logits": _shape(output.predicates.logits),
            "evidence_maps": _shape(output.predicates.evidence_maps),
        },
        "traffic": {
            "factor_names": len(output.traffic.factor_names),
            "lag_weights": _shape(output.traffic.lag_weights),
        },
        "ledger": {"final_prediction_normalized": _shape(output.ledger.final_prediction_normalized)},
        "all_finite": all([
            _finite(output.total_loss),
            _finite(output.backbone.predicate_grid),
            _finite(output.predicates.logits),
            _finite(output.traffic.factor_logits),
            _finite(output.ledger.final_prediction_normalized),
            _finite(output.text.action_logits),
            _finite(output.text.explanation_logits),
        ]),
    }, _finite(output.total_loss) and len(output.predicates.names) == 32 and len(output.traffic.factor_names) == 13)

    reports["video_swin_backbone_audit.json"] = _passed({
        "forward_count": output.backbone.forward_count,
        "predicate_native_time": int(output.backbone.predicate_grid.shape[1]),
        "final_native_time": int(output.backbone.final_grid.shape[1]),
        "dense_final_tokens": _shape(output.backbone.dense_final_tokens),
        "temporal_global": _shape(output.backbone.temporal_global),
        "bf16_or_native_dtype": str(output.backbone.final_grid.dtype),
    }, output.backbone.forward_count == 1 and output.backbone.final_grid.shape[-3:-1] == (7, 7))

    reports["dynamic_predicate_audit.json"] = _passed({
        "predicate_count": len(output.predicates.names),
        "evidence_sum_max_error": evidence_sum_error,
        "temporal_probability_mean_abs_delta": predicate_temporal_delta,
        "confidence_std": float(output.predicates.confidence.detach().float().std().cpu()),
        "relative_motion_shape": _shape(output.predicates.relative_motion),
        "corridor_mass_shape": _shape(output.predicates.corridor_mass),
    }, len(output.predicates.names) == 32 and evidence_sum_error < 1e-3 and predicate_temporal_delta > 0)

    reports["semantic_consolidation_audit.json"] = _passed({
        "slot_names": list(output.semantic_tokens.slot_names),
        "assignment_sum_max_error": _max_abs(dense_assignment_sum - 1.0),
        "conservation_error_max": semantic_mass_error,
        "token_mass_min": float(output.semantic_tokens.token_mass.detach().float().min().cpu()),
        "source_provenance_shape": _shape(output.semantic_tokens.source_provenance),
    }, len(output.semantic_tokens.slot_names) == 5 and _max_abs(dense_assignment_sum - 1.0) < 1e-4 and semantic_mass_error <= 1e-3)

    reports["corridor_flow_audit.json"] = _passed({
        "corridor_shape": _shape(output.traffic.factor_to_corridor),
        "corridor_variance": corridor_variance,
        "lateral_bias_mean_abs": _mean_abs(output.traffic.lateral_bias),
    }, output.traffic.factor_to_corridor.shape[-1] == 3 and corridor_variance > 0)

    reports["pattern_traffic_audit.json"] = _passed({
        "factor_names": list(output.traffic.factor_names),
        "pattern_shape": _shape(output.traffic.pattern_probs),
        "pattern_variance": pattern_variance,
        "factor_token_mean_abs": _mean_abs(output.traffic.factor_tokens_native),
        "lineage_count": len(output.traffic.lineage),
    }, output.traffic.pattern_probs.shape[-1] == 4 and pattern_variance > 0 and len(output.traffic.factor_names) == 13)

    reports["response_lag_audit.json"] = _passed({
        "lag_shape": _shape(output.traffic.lag_weights),
        "lag_sum_max_error": lag_sum_error,
        "lag_temporal_mean_abs_delta": lag_temporal_delta,
    }, output.traffic.lag_weights.shape[-1] == 4 and lag_sum_error < 1e-5 and lag_temporal_delta > 0)

    reports["query_motion_transformer_audit.json"] = _passed({
        "query_hidden": _shape(output.motion.query_hidden),
        "global_prediction_normalized": _shape(output.motion.global_prediction_normalized),
        "source_attention": None if output.motion.source_attention is None else _shape(output.motion.source_attention),
    }, output.motion.query_hidden.shape[1:] == (32, 768) and output.motion.global_prediction_normalized.shape[1:] == (32, 2))

    reports["decision_ledger_audit.json"] = _passed({
        "signal_names": list(output.ledger.signal_names),
        "normalized_reconstruction_max_error": norm_recon_error,
        "raw_reconstruction_max_error": raw_recon_error,
        "speed_attention_shape": _shape(output.ledger.speed_factor_attention),
        "course_attention_shape": _shape(output.ledger.course_factor_attention),
        "benefit_target_present": output.ledger.benefit_target is not None,
    }, tuple(output.ledger.signal_names) == ("course", "speed") and norm_recon_error < 1e-5 and raw_recon_error < 1e-5)

    reports["text_decoder_audit.json"] = _passed({
        "action_loss": float(output.text.action_loss.detach().float().cpu()),
        "explanation_loss": float(output.text.explanation_loss.detach().float().cpu()),
        "generated_action": generated_action,
        "generated_explanation": generated_explanation,
        "action_attention_shape": _shape(output.text.action_to_factor_attention),
        "explanation_attention_shape": _shape(output.text.explanation_to_factor_attention),
        "separate_action_explanation_losses": bool(action_explanation_distinct),
    }, action_explanation_distinct and output.text.action_to_factor_attention.shape[-1] == 13 and output.text.explanation_to_factor_attention.shape[-1] == 13)

    attention = output.text.explanation_to_factor_attention.detach().float().mean(dim=1)
    contribution = output.ledger.gated_factor_contributions_normalized.detach().float().abs().mean(dim=(1, 3))
    if attention.shape == contribution.shape:
        centered_a = attention - attention.mean(dim=-1, keepdim=True)
        centered_c = contribution - contribution.mean(dim=-1, keepdim=True)
        corr = (centered_a * centered_c).sum(dim=-1) / (
            centered_a.square().sum(dim=-1).sqrt() * centered_c.square().sum(dim=-1).sqrt()
        ).clamp_min(1e-6)
        mean_corr = float(corr.mean().cpu())
    else:
        mean_corr = 0.0
    reports["gradient_direction_audit.json"] = _passed({
        "explanation_attention_vs_contribution_corr": mean_corr,
        "gradient_abs_sum": gradient_abs_sum,
        "nonzero_groups": sorted(grad_nonzero),
    }, len(grad_nonzero) >= 3)

    reports["loss_audit.json"] = _passed({
        "loss_components": loss_components,
        "nonzero_loss_names": nonzero_loss_names,
        "has_raw_predicate_nnpu": "raw/predicate_nnpu" in loss_components,
        "configured_weights": output.diagnostics.get("loss_weights", {}),
    }, "predicate_nnpu" in nonzero_loss_names and "action_text" in nonzero_loss_names and "explanation_text" in nonzero_loss_names)

    reports["gate_gradient_chain.json"] = _passed({
        "gradient_abs_sum": gradient_abs_sum,
        "nonzero_groups": sorted(grad_nonzero),
    }, len(grad_nonzero) >= 3)

    reports["gate_identity_checks.json"] = _passed({
        "semantic_conservation_error_max": semantic_mass_error,
        "normalized_ledger_error_max": norm_recon_error,
        "raw_ledger_error_max": raw_recon_error,
    }, semantic_mass_error <= 1e-3 and norm_recon_error < 1e-5 and raw_recon_error < 1e-5)

    reports["gate_temporal_lag.json"] = _passed({
        "lag_sum_max_error": lag_sum_error,
        "lag_temporal_mean_abs_delta": lag_temporal_delta,
        "pattern_temporal_mean_abs_delta": _mean_abs(output.traffic.pattern_probs[:, 1:] - output.traffic.pattern_probs[:, :-1]),
    }, lag_sum_error < 1e-5 and lag_temporal_delta > 0)

    return reports


def audit_oia(cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    paths = cfg["paths"]
    name_features = load_bert_name_features(paths["bert_dir"])
    module = PredicateQueryTransfer(
        dim=int(cfg["model"]["dimensions"]["predicate"]),
        gate_init=float(cfg["model"]["oia_transfer"]["transfer_gate_init"]),
        name_features=name_features,
    )
    module.load_oia_query(
        paths["oia_acpr_checkpoint"],
        cfg["model"]["oia_transfer"]["tensor_key"],
    )
    baseline, source = module()
    with torch.no_grad():
        module.name_features.add_(0.01)
    name_changed, _ = module()
    with torch.no_grad():
        module.name_features.sub_(0.01)
        module.oia_query.add_(0.01)
    oia_changed, _ = module()
    with torch.no_grad():
        module.oia_query.sub_(0.01)
        module.domain_residual.add_(0.01)
    residual_changed, _ = module()
    deltas = {
        "name": float((baseline - name_changed).abs().mean()),
        "oia": float((baseline - oia_changed).abs().mean()),
        "residual": float((baseline - residual_changed).abs().mean()),
    }
    passed = source.get("loaded") is True and all(value > 0 for value in deltas.values())
    return _write(output_dir / "oia_transfer_audit.json", {
        "status": "pass" if passed else "blocked",
        "passed": passed,
        "source": source,
        "name_feature_shape": list(name_features.shape),
        "query_shape": list(baseline.shape),
        "perturbation_mean_abs_delta": deltas,
    })


def audit_real_batch(
    cfg: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    scan_samples: int,
) -> dict[str, Any]:
    loader = build_dynflow_swin_dataloader(
        cfg, split="train", batch_size=1, max_samples=scan_samples, synthetic=False
    )
    labeler = PredicateRuleLabeler.from_yaml(cfg["model"]["nnpu"]["rules_yaml"])
    counts = {"positive": 0, "reliable_negative": 0, "unlabeled": 0}
    first_batch = None
    scanned = 0
    for batch in loader:
        if first_batch is None:
            first_batch = batch
        labels = labeler.label(
            [f"{a} {r}" for a, r in zip(batch.raw_actions, batch.raw_justifications)]
        )
        counts["positive"] += int(labels.positive.sum())
        counts["reliable_negative"] += int(labels.reliable_negative.sum())
        counts["unlabeled"] += int(labels.unlabeled.sum())
        scanned += len(batch.sample_ids)
        if scanned >= scan_samples:
            break
    if first_batch is None:
        raise RuntimeError("real train loader produced no batch")
    model = ACPRDynFlowSwinModel(cfg).to(device)
    batch = move_batch_to_device(first_batch, device)
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(batch)
    output.total_loss.backward()
    gradient_abs_sum: dict[str, float] = {}
    gradient_prefixes = {
        "backbone": ("backbone",),
        "predicates": ("predicates", "predicate_transfer"),
        "traffic": ("traffic", "corridor", "pattern"),
        "motion": ("motion",),
        "ledger": ("ledger",),
        "text": ("text",),
    }
    for group, prefixes in gradient_prefixes.items():
        total = 0.0
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if any(name.startswith(prefix) or f".{prefix}" in name for prefix in prefixes):
                total += float(parameter.grad.detach().abs().sum().cpu())
        gradient_abs_sum[group] = total
    predicate_grad = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for name, parameter in model.named_parameters()
        if ("predicate" in name or "calalign" in name) and parameter.grad is not None
    )
    loss_value = float(output.loss_components["predicate_nnpu"].detach().cpu())
    passed = all(value > 0 for value in counts.values()) and loss_value > 0 and predicate_grad > 0
    nnpu_report = _write(output_dir / "nnpu_calalign_audit.json", {
        "status": "pass" if passed else "blocked",
        "passed": passed,
        "scan_samples": scanned,
        "counts": counts,
        "batch_counts": output.diagnostics["nnpu_counts"],
        "predicate_nnpu_loss": loss_value,
        "predicate_grad_abs_sum": predicate_grad,
        "calalign_update_count": output.diagnostics["calalign_update_count"],
    })
    runtime_reports = build_runtime_gate_reports(output, gradient_abs_sum)
    for report_name, payload in runtime_reports.items():
        _write(output_dir / report_name, payload)

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        base = model(batch)
        reverse = run_intervention(model, batch, InterventionSpec("temporal_reverse"))
        factor = run_intervention(model, batch, InterventionSpec("factor_off", factor_index=0))
    deltas = {
        "temporal_reverse_control": float(
            (base.ledger.final_prediction_normalized - reverse.ledger.final_prediction_normalized)
            .abs().mean().cpu()
        ),
        "factor_off_control": float(
            (base.ledger.final_prediction_normalized - factor.ledger.final_prediction_normalized)
            .abs().mean().cpu()
        ),
        "temporal_reverse_text": float(
            (base.text.explanation_logits - reverse.text.explanation_logits).abs().mean().cpu()
        ),
        "factor_off_text": float(
            (base.text.explanation_logits - factor.text.explanation_logits).abs().mean().cpu()
        ),
    }
    intervention_passed = all(value > 0 for value in deltas.values())
    intervention_report = _write(output_dir / "intervention_audit.json", {
        "status": "pass" if intervention_passed else "blocked",
        "passed": intervention_passed,
        "deltas": deltas,
        "traces": {
            "temporal_reverse": reverse.diagnostics["intervention"],
            "factor_off": factor.diagnostics["intervention"],
        },
    })
    git_sha = _git(["rev-parse", "HEAD"])
    config_hash = _sha("configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    checkpoint_hash = "untrained_preflight_model"
    case = tensor_case_from_output(
        base,
        sample_id=batch.sample_ids[0],
        git_sha=git_sha,
        config_hash=config_hash,
        checkpoint_hash=checkpoint_hash,
        counterfactuals=deltas,
    )
    visual_dir = output_dir / "visuals"
    canvas = render_case_canvas(case, visual_dir / "case.png", visual_dir / "case.json")
    atlas = build_atlas([canvas], visual_dir / "atlas.html", visual_dir / "atlas.json")
    visual_report = _write(output_dir / "visual_artifact_index.json", {
        "status": "pass",
        "passed": True,
        "canvas_png": str(visual_dir / "case.png"),
        "canvas_json": str(visual_dir / "case.json"),
        "atlas_html": str(visual_dir / "atlas.html"),
        "atlas_json": str(visual_dir / "atlas.json"),
        "tensor_sources": atlas["tensor_sources"],
    })
    return {
        "nnpu": nnpu_report,
        "intervention": intervention_report,
        "visual": visual_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scan_samples", type=int, default=128)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    report = {
        "oia": audit_oia(cfg, output_dir),
        "real_batch": audit_real_batch(cfg, output_dir, torch.device(args.device), args.scan_samples),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
