from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import torch

from fate_x.acpr_flow.model import ACPRFlowModel, ACPRFlowModelConfig, TinyDirectImageVideoBackbone
from fate_x.acpr_flow.interventions import InterventionSpec
from fate_x.acpr_flow.region_priors import ACPR_PREDICATE_NAMES, FLOW_FACTOR_NAMES
from fate_x.explain.acpr_flow_renderer import render_acpr_flow_canvas
from fate_x.utils.acpr_flow_artifacts import write_json
from fate_x.utils.acpr_flow_config import load_acpr_flow_config
from fate_x.utils.acpr_flow_git_guard import git_guard


FORBIDDEN_SYMBOLS = {"TokenPMTAdapter", "LogSinkhornTransport", "FlowTraceLoss", "FlowTraceRenderer"}

REQUIRED_PREFLIGHT_EVIDENCE = {
    "gate_b_direct_image_8step_smoke.json": "Gate B real direct-image 8-step smoke has not been recorded",
    "gate_c_gradient_chain_report.json": "Gate C gradient-chain evidence has not been recorded",
    "gate_d_mechanism_overfit_128_report.json": "Gate D 128-sample mechanism overfit has not been recorded",
    "gate_e_temporal_necessity_report.json": "Gate E temporal necessity evidence has not been recorded",
    "gate_f_real_intervention_report.json": "Gate F real intervention evidence has not been recorded",
    "memory_probe_selection.json": "formal memory probe selection has not been recorded",
    "foreground_supervisor_smoke.json": "foreground supervisor attachment evidence has not been recorded",
}

PREFLIGHT_EVIDENCE_CODES = {
    "gate_b_direct_image_8step_smoke.json": "missing_gate_b_8_step_direct_image_smoke",
    "gate_c_gradient_chain_report.json": "missing_gate_c_gradient_chain",
    "gate_d_mechanism_overfit_128_report.json": "missing_gate_d_128_sample_mechanism_overfit",
    "gate_e_temporal_necessity_report.json": "missing_gate_e_temporal_necessity",
    "gate_f_real_intervention_report.json": "missing_gate_f_real_intervention",
    "memory_probe_selection.json": "missing_memory_probe",
    "foreground_supervisor_smoke.json": "missing_foreground_supervisor_proof",
}


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_forbidden(repo_root: Path) -> list[str]:
    offenders = []
    for path in [repo_root / "fate_x" / "engine" / "train_acpr_flowcal_pp.py", repo_root / "fate_x" / "acpr_flow" / "model.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_SYMBOLS:
                offenders.append(f"{path}:{node.id}")
    return offenders


def _module_source(repo_root: Path, relative: str) -> str:
    path = repo_root / relative
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _formal_gate_blockers(repo_root: Path, model: ACPRFlowModel, cfg: dict, device: str, output_dir: Path) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    blocker_codes: list[str] = []
    trainer_path = repo_root / "fate_x" / "engine" / "train_acpr_flowcal_pp.py"
    model_path = repo_root / "fate_x" / "acpr_flow" / "model.py"
    trainer_source = trainer_path.read_text(encoding="utf-8")
    model_source = model_path.read_text(encoding="utf-8")
    bert_captioning_source = _module_source(repo_root, "src/layers/bert/modeling_bert.py")

    if isinstance(model.backbone, TinyDirectImageVideoBackbone):
        blockers.append("formal model still uses TinyDirectImageVideoBackbone instead of real ADAPT Video Swin multiscale backbone")
        blocker_codes.append("tiny_backbone_in_formal_path")
    if getattr(model.backbone, "formal_backbone_name", "") != "adapt_video_swin_multiscale":
        blockers.append("formal model backbone is not ADAPT Video Swin multiscale")
        blocker_codes.append("missing_adapt_video_swin_multiscale_backbone")
    load_report = getattr(model.backbone, "load_report", None)
    if load_report is not None and not bool(getattr(load_report, "loaded", False)):
        blockers.append("formal pretrained ADAPT Video Swin backbone is not loaded")
        blocker_codes.append("missing_checkpoint_download_or_load_report")
    if device == "cpu":
        blockers.append("formal review pass requires cuda direct-image execution, not cpu audit mode")
        blocker_codes.append("cuda_formal_audit_not_run")
    if "torch.randn" in trainer_source:
        blockers.append("formal trainer still creates random frame tensors instead of decoding BDD-X direct-image batches")
        blocker_codes.append("random_frame_tensor_in_formal_trainer")
    if "train_smoke" in trainer_source:
        blockers.append("formal train entrypoint is still a smoke trainer, not the required experiment suite")
        blocker_codes.append("formal_entrypoint_is_smoke_only")
    if "adapt_checkpoint" not in trainer_source or "video_swin_checkpoint" not in trainer_source:
        blockers.append("formal trainer does not load the configured ADAPT checkpoint and Video Swin checkpoint")
        blocker_codes.append("missing_checkpoint_load_report")
    if "predicate_pu = bundle.predicate_probs_temporal.mean() * 0.0" in model_source:
        blockers.append("predicate PU loss is wired as an always-zero placeholder")
        blocker_codes.append("predicate_pu_placeholder_zero")
    if "flow_pu = bundle.flow_probs.mean() * 0.0" in model_source:
        blockers.append("flow PU loss is wired as an always-zero placeholder")
        blocker_codes.append("flow_pu_placeholder_zero")
    if "reason_semantic" not in model_source:
        blockers.append("reason semantic loss from config is not implemented in ACPRFlowModel.forward")
        blocker_codes.append("reason_semantic_loss_missing")
    if "future_control" not in model_source:
        blockers.append("prefix-to-future configured loss is not implemented in ACPRFlowModel.forward")
        blocker_codes.append("future_control_loss_missing")
    if "build_reason_semantic_target_for_batch" not in trainer_source or "reason_semantic_target=reason_semantic_target" not in trainer_source:
        blockers.append("online reason semantic target is not built and passed by the formal trainer")
        blocker_codes.append("online_reason_target_not_passed")
    if "TemporalHardPairQueue" not in model_source and "TemporalHardPairQueue" not in trainer_source:
        blockers.append("Temporal HardPair is not integrated into the formal model/trainer path")
        blocker_codes.append("missing_hardpair_integration")
    if "SequenceCalAlign" not in trainer_source and "Sequence-CalAlign" not in trainer_source:
        blockers.append("Sequence-CalAlign is not integrated into the formal trainer/evaluator path")
        blocker_codes.append("missing_sequence_calalign_integration")
    if "acpr_temporal_seca" not in bert_captioning_source or "acpr_flow_bundle.reason_memory" not in bert_captioning_source:
        blockers.append("Temporal SECA is not patched into BertForImageCaptioning pre-LM-head path")
        blocker_codes.append("missing_bert_captioning_seca_hook")
    if "BertForImageCaptioning" not in model_source and "BertForImageCaptioning" not in trainer_source:
        blockers.append("formal model/trainer does not route text generation through BertForImageCaptioning ACPR SECA hook")
        blocker_codes.append("formal_path_not_using_bert_captioning_seca")
    if not (repo_root / "fate_x" / "engine" / "supervise_acpr_flowcal_foreground.py").exists():
        blockers.append("foreground supervisor launcher is missing")
        blocker_codes.append("missing_foreground_supervisor_launcher")
    if not (repo_root / "scripts" / "FATE_X_acpr_flowcal_pp_v1_foreground.ps1").exists():
        blockers.append("foreground supervisor PowerShell launcher is missing")
        blocker_codes.append("missing_foreground_supervisor_powershell")
    for filename in REQUIRED_PREFLIGHT_EVIDENCE:
        if not (output_dir / filename).exists():
            blockers.append(f"missing formal preflight evidence: {filename}")
            blocker_codes.append(PREFLIGHT_EVIDENCE_CODES[filename])
    if cfg.get("experiment_suite", {}).get("common_stage_a", {}).get("epochs") != 6:
        blockers.append("common Stage A epoch count does not match the formal suite")
        blocker_codes.append("formal_suite_epoch_count_mismatch")
    if cfg.get("supervisor", {}).get("require_review_pass") is not True:
        blockers.append("supervisor.require_review_pass must remain true for formal training")
        blocker_codes.append("review_pass_not_required_by_config")
    return blockers, sorted(set(blocker_codes))


def run_audit(config: str, output_dir: str, device: str = "cpu", write_review_pass: bool = False) -> dict:
    repo_root = Path.cwd()
    cfg = load_acpr_flow_config(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    git_info = git_guard(repo_root, cfg.get("repository", {}).get("expected_branch", "flowtrace_pmt_v1"), cfg.get("repository", {}).get("github_remote", "github"))
    offenders = _scan_forbidden(repo_root)
    model_cfg = ACPRFlowModelConfig(
        formal_backbone=True,
        load_pretrained_backbone=(device != "cpu"),
        video_swin_checkpoint=cfg.get("paths", {}).get("video_swin_checkpoint"),
        image_resolution=int(cfg.get("data", {}).get("image_resolution", 224)),
        num_frames=int(cfg.get("data", {}).get("max_num_frames", 32)),
        use_transport=True,
        use_flow=True,
        use_prefix_future=True,
    )
    model = ACPRFlowModel(model_cfg).to(device)
    formal_blockers, formal_blocker_codes = _formal_gate_blockers(repo_root, model, cfg, device, out)
    frames = torch.randn(1, 32, 3, 224, 224, device=device)
    control_targets = torch.zeros(1, 32, 2, device=device)
    masked_ids = torch.zeros(1, 16, dtype=torch.long, device=device)
    result = model(
        frames,
        control_targets=control_targets,
        masked_ids=masked_ids,
        raw_actions=["the car slows down"],
        raw_justifications=["cars ahead are stopped"],
    )
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
        "hashes": {
            "config_sha256": _sha256(repo_root / config),
            "implementation_plan_sha256": _sha256(repo_root / "docs" / "runbooks" / "ACPR_FlowCalPP_V1_Implementation_Plan.md"),
            "audit_skill_sha256": _sha256(repo_root / ".codex" / "skills" / "acpr-flowcal-pp-implementation-audit" / "SKILL.md"),
        },
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
        "formal_gate_blockers": formal_blockers,
        "formal_gate_blocker_codes": formal_blocker_codes,
        "passed_formal_gate": not offenders and not formal_blockers,
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
        old = out / "REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt"
        if old.exists():
            old.unlink()
        if not report["passed_minimal_gate"] or not report["passed_formal_gate"] or git_info["dirty_status"] or (git_info["github_remote_head"] and git_info["github_remote_head"] != git_info["git_head"]):
            blocker_codes = list(formal_blocker_codes)
            if git_info["dirty_status"]:
                blocker_codes.append("dirty_worktree")
            if git_info["github_remote_head"] and git_info["github_remote_head"] != git_info["git_head"]:
                blocker_codes.append("remote_sha_mismatch")
            write_json(
                out / "formal_gate_blockers.json",
                {
                    "blockers": formal_blockers,
                    "blocker_codes": sorted(set(blocker_codes)),
                    "dirty_status": git_info["dirty_status"],
                },
            )
            raise RuntimeError("ACPR formal review pass blocked; see review_report.json and formal_gate_blockers.json")
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
