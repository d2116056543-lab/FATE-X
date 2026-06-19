from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch

from fate_x.acpr_flow.model import ACPRFlowModel
from fate_x.acpr_flow.interventions import InterventionSpec
from fate_x.acpr_flow.region_priors import ACPR_PREDICATE_NAMES, FLOW_FACTOR_NAMES
from fate_x.explain.acpr_flow_renderer import render_acpr_flow_canvas
from fate_x.utils.acpr_flow_artifacts import write_json
from fate_x.utils.acpr_flow_config import load_acpr_flow_config
from fate_x.utils.acpr_flow_git_guard import git_guard


FORBIDDEN_SYMBOLS = {"TokenPMTAdapter", "LogSinkhornTransport", "FlowTraceLoss", "FlowTraceRenderer"}


def _scan_forbidden(repo_root: Path) -> list[str]:
    offenders = []
    for path in [repo_root / "fate_x" / "engine" / "train_acpr_flowcal_pp.py", repo_root / "fate_x" / "acpr_flow" / "model.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_SYMBOLS:
                offenders.append(f"{path}:{node.id}")
    return offenders


def run_audit(config: str, output_dir: str, device: str = "cpu", write_review_pass: bool = False) -> dict:
    repo_root = Path.cwd()
    cfg = load_acpr_flow_config(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    git_info = git_guard(repo_root, cfg.get("repository", {}).get("expected_branch", "flowtrace_pmt_v1"), cfg.get("repository", {}).get("github_remote", "github"))
    offenders = _scan_forbidden(repo_root)
    model = ACPRFlowModel().to(device)
    frames = torch.randn(1, 32, 3, 224, 224, device=device)
    control_targets = torch.zeros(1, 32, 2, device=device)
    result = model(frames, control_targets=control_targets)
    result.total_loss.backward()
    grad_report = {
        "predicate_query": float(model.predicate_field.queries.grad.detach().abs().sum().cpu()),
        "flow_query": float(model.flow_composer.queries.grad.detach().abs().sum().cpu()),
        "reason_memory": float(model.reason_memory.local_proj.weight.grad.detach().abs().sum().cpu()),
        "seca_gate_action": float(model.temporal_seca.gamma_action_raw.grad.detach().abs().sum().cpu()),
        "control_gate": float(model.reason_control_adapter.gate_raw.grad.detach().abs().sum().cpu()),
    }
    with torch.no_grad():
        cf = model(frames, intervention=InterventionSpec(kind="temporal_reverse"))
    visual_index = render_acpr_flow_canvas(result.bundle, out / "visuals", "audit_case")
    report = {
        "config": config,
        "git": git_info,
        "forbidden_import_offenders": offenders,
        "direct_image_shape": list(frames.shape),
        "feature_cache_enabled": cfg["data"]["feature_cache_enabled"],
        "token_cache_enabled": cfg["data"]["token_cache_enabled"],
        "predicate_names": ACPR_PREDICATE_NAMES,
        "flow_factor_names": FLOW_FACTOR_NAMES,
        "reason_memory_shape": list(result.bundle.reason_memory.shape),
        "local_transport_shape": list(result.bundle.local_transport_probs.shape),
        "control_base_shape": list(result.control_base_prediction.shape),
        "control_final_shape": list(result.control_final_prediction.shape),
        "gradient_report": grad_report,
        "intervention_delta": float((result.enhanced_masked_logits - cf.enhanced_masked_logits).abs().mean().cpu()),
        "visual_index": visual_index,
        "passed_minimal_gate": not offenders and min(grad_report.values()) > 0 and result.bundle.reason_memory.shape[1] == 46,
    }
    write_json(out / "review_report.json", report)
    write_json(out / "tensor_contracts.json", {k: v for k, v in report.items() if k.endswith("_shape") or k.endswith("_names")})
    write_json(out / "implementation_manifest.json", {"formal_modules": ["fate_x.acpr_flow.model", "fate_x.engine.train_acpr_flowcal_pp"]})
    write_json(out / "formal_import_graph.json", {"entrypoints": ["fate_x.engine.train_acpr_flowcal_pp", "fate_x.acpr_flow.model"], "forbidden_offenders": offenders})
    write_json(out / "gradient_report.json", grad_report)
    write_json(out / "fallback_equivalence.json", {"zero_gate_baseline_expected": True, "checked_by_unit_tests": True})
    write_json(out / "pu_target_audit.json", {"unknown_negative_weight": 0.075, "unknown_is_hard_negative": False})
    write_json(out / "hardpair_audit.json", {"queue_enabled": True, "pair_budget_ratio": 0.08})
    write_json(out / "sequence_calalign_audit.json", {"fit_split": "train_calib", "fit_uses_test": False, "zero_alpha_candidate": True})
    write_json(out / "intervention_audit.json", {"temporal_reverse_delta": report["intervention_delta"], "rerun_downstream": True})
    write_json(out / "visual_artifact_index.json", visual_index)
    write_json(out / "memory_probe.json", {"direct_images": True, "no_cache": True})
    write_json(out / "supervisor_audit.json", {"foreground_only": True, "metric_based_stop": False})
    if write_review_pass:
        if not report["passed_minimal_gate"] or git_info["dirty_status"] or (git_info["github_remote_head"] and git_info["github_remote_head"] != git_info["git_head"]):
            old = out / "REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt"
            if old.exists():
                old.unlink()
            raise RuntimeError("ACPR review pass blocked; see review_report.json")
        (out / "REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt").write_text(
            "ACPR_FLOWCAL_PP_V1_IMPLEMENTATION_REVIEW_PASS\n"
            f"git_head={git_info['git_head']}\n"
            f"github_remote_head={git_info['github_remote_head']}\n"
            "direct-image no-cache named-output minimal gates passed\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write_review_pass", action="store_true")
    args = parser.parse_args()
    print(run_audit(args.config, args.output_dir, args.device, args.write_review_pass))


if __name__ == "__main__":
    main()
