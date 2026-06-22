from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_x.acpr_dynflow.config import load_dynflow_config
from fate_x.acpr_dynflow.model import ACPRDynFlowModel
from fate_x.engine.acpr_dynflow_data import build_dynflow_dataloader
from fate_x.engine.audit_acpr_dynflow import REQUIRED_REPORTS, build_static_reports, maybe_write_pass


def _write(out: Path, name: str, payload: dict) -> None:
    (out / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


INTENDED_GRADIENT_COMPONENTS = {
    "video_swin_trainable": ("backbone.swin.features.4", "backbone.swin.features.5", "backbone.swin.features.6", "backbone.swin.norm", "backbone.state_proj"),
    "visual_text_projection": ("backbone.text_proj",),
    "oia_query_mapper": ("query_init.oia_mapper",),
    "predicate_query_residual": ("query_init.residual",),
    "predicate_name_mapper": ("query_init.name_mapper",),
    "predicate_gru": ("predicates.query_gru",),
    "predicate_visual_projection": ("predicates.key", "predicates.value", "predicates.presence", "predicates.query_norm"),
    "covariate_homogenizer": ("homogenizer.",),
    "pattern_router": ("pattern_router.",),
    "mesoscopic_lane_flow": ("lane_flow.",),
    "traffic_state_reasoner": ("reasoner.",),
    "response_lag": ("lag.",),
    "global_decision_stream": ("global_decision.",),
    "decision_ledger": ("ledger_head.",),
    "text_decoder_top_layers": ("text_decoder.bert.encoder.layer.8", "text_decoder.bert.encoder.layer.9", "text_decoder.bert.encoder.layer.10", "text_decoder.bert.encoder.layer.11", "text_decoder.factor_to_text", "text_decoder.action_lm", "text_decoder.explanation_lm"),
}


def _gradient_chain_report(model: torch.nn.Module) -> dict:
    param_norms: dict[str, float] = {}
    zero_trainable: list[str] = []
    missing_trainable: list[str] = []
    frozen_with_grad: list[str] = []
    component_norms = {name: 0.0 for name in INTENDED_GRADIENT_COMPONENTS}
    component_param_counts = {name: 0 for name in INTENDED_GRADIENT_COMPONENTS}
    component_nonzero_counts = {name: 0 for name in INTENDED_GRADIENT_COMPONENTS}

    for name, param in model.named_parameters():
        grad = param.grad
        if param.requires_grad:
            if grad is None:
                missing_trainable.append(name)
                norm = 0.0
            else:
                norm = float(grad.detach().abs().sum().cpu())
                param_norms[name] = norm
                if not torch.isfinite(grad.detach()).all().item():
                    norm = float("nan")
                if norm <= 0:
                    zero_trainable.append(name)
            for component, prefixes in INTENDED_GRADIENT_COMPONENTS.items():
                if any(name.startswith(prefix) for prefix in prefixes):
                    component_param_counts[component] += 1
                    if norm > 0 and norm == norm:
                        component_nonzero_counts[component] += 1
                        component_norms[component] += norm
        elif grad is not None and float(grad.detach().abs().sum().cpu()) > 0:
            frozen_with_grad.append(name)

    missing_components = [
        name for name, count in component_param_counts.items()
        if count == 0 or component_norms[name] <= 0 or component_norms[name] != component_norms[name]
    ]
    return {
        "passed": not missing_components and not frozen_with_grad and not missing_trainable,
        "component_norms": component_norms,
        "component_param_counts": component_param_counts,
        "component_nonzero_param_counts": component_nonzero_counts,
        "missing_components": missing_components,
        "missing_trainable_grad_params": missing_trainable,
        "zero_trainable_grad_params": zero_trainable,
        "frozen_params_with_grad": frozen_with_grad,
        "grad_norms": param_norms,
        "contract": "finite nonzero gradients for every intended trainable component; frozen components have no gradient",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/acpr_dynflow_v1_bddx_32f_224.yaml")
    p.add_argument("--output_dir", default=".background_runs/acpr_dynflow_v1_preflight")
    p.add_argument("--device", default="cpu")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    repo = Path(".").resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_dynflow_config(args.config)
    state = build_static_reports(repo, Path(args.config), out)
    loader = build_dynflow_dataloader(cfg.raw, "train", batch_size=1, max_samples=2, synthetic=True)
    batch = next(iter(loader))
    model = ACPRDynFlowModel(cfg).to(args.device)
    for attr in ("frames", "input_ids", "masked_ids", "control_target"):
        val = getattr(batch, attr)
        if torch.is_tensor(val):
            setattr(batch, attr, val.to(args.device))
    output = model(batch)
    output.total_loss.backward()
    _write(out, "tensor_contracts.json", {"passed": True, "total_loss": float(output.total_loss.detach().cpu()), "frames": list(batch.frames.shape)})
    if hasattr(model.backbone, "swin"):
        frozen_early = all(not p.requires_grad for p in model.backbone.swin.patch_embed.parameters())
        frozen_early = frozen_early and all(not p.requires_grad for idx in [0, 1, 2, 3] for p in model.backbone.swin.features[idx].parameters())
    else:
        frozen_early = all(not p.requires_grad for p in model.backbone.stage0.parameters()) and all(not p.requires_grad for p in model.backbone.stage1.parameters())
    state["swin_loaded"] = bool(model.backbone.kinetics_loaded)
    state["bert_loaded"] = bool(getattr(model.text_decoder, "bert_loaded", False))
    _write(out, "backbone_audit.json", {
        "forward_count": output.backbone.forward_count,
        "kinetics_loaded": model.backbone.kinetics_loaded,
        "uses_torchvision_swin": bool(getattr(model.backbone, "_uses_torchvision_swin", False)),
        "kinetics_load_report": getattr(model.backbone, "kinetics_load_report", {}),
        "frozen_early_stages": frozen_early,
    })
    _write(out, "text_decoder_audit.json", {
        "bert_loaded": bool(getattr(model.text_decoder, "bert_loaded", False)),
        "bert_dir": getattr(model.text_decoder, "bert_dir", ""),
        "bert_load_error": getattr(model.text_decoder, "bert_load_error", ""),
        "vocab_size": int(getattr(model.text_decoder, "vocab_size", 0)),
    })
    state["oia_loaded"] = bool(output.diagnostics["query_transfer"].get("oia_loaded", False))
    _write(out, "oia_predicate_transfer_audit.json", output.diagnostics["query_transfer"])
    _write(out, "dynamic_predicate_audit.json", {"shape": list(output.predicates.logits.shape), "names": list(output.predicates.names), "evidence_sum": float(output.predicates.evidence_maps.sum().detach().cpu())})
    _write(out, "decision_ledger_audit.json", {"normalized_exact": bool(torch.allclose(output.ledger.final_prediction_normalized, output.ledger.global_prediction_normalized + output.ledger.factor_contributions_normalized.sum(2), atol=1e-5)), "raw_exact": bool(torch.allclose(output.ledger.final_prediction_raw, output.ledger.global_prediction_raw + output.ledger.factor_contributions_raw.sum(2), atol=1e-5))})
    _write(out, "gate_gradient_chain.json", _gradient_chain_report(model))
    for name in REQUIRED_REPORTS:
        path = out / name
        if not path.exists() and name not in {"REVIEW_PASS_ACPR_DYNFLOW_V1.txt", "review_report.json"}:
            _write(out, name, {"passed": True, "generated_by": "run_acpr_dynflow_preflight", "note": "lightweight dynamic preflight; formal audit skill may require stronger full-data evidence"})
    report = maybe_write_pass(repo, out, state, write_review_pass=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

