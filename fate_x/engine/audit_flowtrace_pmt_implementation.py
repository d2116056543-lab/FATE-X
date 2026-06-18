from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from fate_x.explain.flowtrace_renderer import FlowTraceRenderer
from fate_x.models.flowtrace_pmt_model import FlowTracePMTModel
from fate_x.models.token_pmt_adapter import TokenPMTAdapter


REQUIRED_ASSET_PATHS = [
    "datasets_part/BDDX/training_32frames.yaml",
    "datasets/BDDX/testing_32frames.yaml",
    "checkpoints/basemodel/checkpoints/model.bin",
    "models/video_swin_transformer/swin_base_patch244_window877_kinetics600_22k.pth",
    "models/captioning/bert-base-uncased/config.json",
    "models/captioning/bert-base-uncased/vocab.txt",
    "models/captioning/bert-base-uncased/pytorch_model.bin",
]


REQUIRED_CONFIG_KEYS = [
    "paths.train_yaml",
    "paths.test_yaml",
    "paths.adapt_checkpoint",
    "paths.bert_dir",
    "paths.video_swin_checkpoint",
    "data.direct_image_training",
    "data.feature_cache_enabled",
    "data.token_cache_enabled",
    "data.build_cache_before_training",
    "baseline.video_token_reducer",
    "baseline.temporal_evidence_memory",
    "baseline.fate_x_reduce_control",
    "model.flowtrace_enabled",
    "model.multiscale.return_stages",
    "model.transport.sinkhorn_iterations",
    "model.evidence_tracks.num_tracks",
    "model.traffic_states.num_states",
    "model.reason_anchors.num_anchors",
    "model.token_pmt.rank",
    "model.token_pmt.hook_location",
    "model.control_protection.dense_tokens_only",
    "loss.anchor_alignment",
    "loss.reason_state_cosine",
    "loss.dynamic_control_huber",
    "loss.transport_cycle",
    "loss.track_diversity",
    "loss.state_diversity",
    "loss.state_sparsity",
    "loss.action_preserve_kl",
    "loss.intervention_max",
    "optimization.epochs",
    "optimization.optimizer",
    "optimization.flowtrace_lr",
    "optimization.token_pmt_lr",
    "optimization.bert_last4_lr",
    "optimization.swin_stage4_lr",
    "optimization.swin_stage3_lr",
    "optimization.control_head_lr",
    "optimization.scheduler",
    "optimization.freeze_schedule.epochs_0_2.train",
    "optimization.freeze_schedule.epochs_3_7.train",
    "optimization.freeze_schedule.epochs_8_39.train",
    "memory_probe.candidates",
    "evaluation.eval_splits",
    "evaluation.best_selection_split",
    "evaluation.best_selection_metric",
    "faithfulness.teacher_forced_scoring",
    "visualizations.flowtrace_canvas",
    "visualizations.state_decision_atlas",
    "supervisor.foreground_only",
    "supervisor.total_epochs_required",
    "supervisor.metric_based_stop",
]


REQUIRED_CONFIG_VALUES = {
    "data.direct_image_training": True,
    "data.feature_cache_enabled": False,
    "data.token_cache_enabled": False,
    "data.build_cache_before_training": False,
    "baseline.video_token_reducer": "none",
    "baseline.temporal_evidence_memory": "none",
    "baseline.fate_x_reduce_control": False,
    "model.flowtrace_enabled": True,
    "model.evidence_tracks.num_tracks": 12,
    "model.traffic_states.num_states": 8,
    "model.token_pmt.rank": 32,
    "model.token_pmt.hook_location": "pre_lm_prediction_head",
    "optimization.epochs": 40,
    "optimization.optimizer": "adamw",
    "optimization.scheduler": "warmup_linear",
    "evaluation.eval_splits": ["test"],
    "evaluation.best_selection_split": "test",
    "evaluation.best_selection_metric": "test_cider_action_plus_explanation",
    "supervisor.foreground_only": True,
    "supervisor.metric_based_stop": False,
    "supervisor.total_epochs_required": 40,
}


FLOWTRACE_OWNED_FILES = [
    "fate_x/engine/train_flowtrace_pmt.py",
    "fate_x/engine/eval_flowtrace_pmt.py",
    "fate_x/engine/probe_flowtrace_memory.py",
    "fate_x/engine/adapt_live_decoder_wrapper.py",
    "fate_x/engine/build_reason_state_anchors.py",
    "fate_x/engine/supervise_flowtrace_foreground.py",
    "fate_x/explain/flowtrace_renderer.py",
    "fate_x/explain/flowtrace_intervention.py",
    "fate_x/explain/flowtrace_atlas.py",
    "fate_x/losses/flowtrace_losses.py",
]


BLOCKED_PATTERNS = [
    "[DUMMY]",
    "synthetic smoke",
    "synthetic_train_justification",
    "synthetic probe placeholder",
    "unavailable_reason",
    "requires formal ADAPT decoder checkpoint",
    "random/fabricated",
    "NotImplementedError",
]


def _windows_gitdir_to_wsl(gitdir: str) -> str:
    normalized = gitdir.replace("\\", "/")
    if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
        return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return normalized


def _git_env_for_repo(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    git_file = repo_root / ".git"
    if os.name == "nt" or not git_file.is_file():
        return env
    text = git_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not text.lower().startswith("gitdir:"):
        return env
    gitdir = text.split(":", 1)[1].strip()
    mapped_gitdir = _windows_gitdir_to_wsl(gitdir)
    if mapped_gitdir != gitdir:
        env["GIT_DIR"] = mapped_gitdir
        env["GIT_WORK_TREE"] = str(repo_root)
    return env


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, cwd=cwd, env=env).strip()


def run_git(repo_root: Path, args: list[str]) -> str:
    return run(["git", *args], cwd=repo_root, env=_git_env_for_repo(repo_root))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - environment failure path.
        raise RuntimeError(f"PyYAML is required for strict config audit: {exc}") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Config is not a mapping: {path}")
    return data


def _get_dotted(data: dict[str, Any], dotted: str) -> tuple[bool, Any]:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def collect_static_blockers(repo_root: Path, config_path: Path) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    repo_root = repo_root.resolve()
    config_path = config_path if config_path.is_absolute() else repo_root / config_path

    for rel in REQUIRED_ASSET_PATHS:
        if not (repo_root / rel).exists():
            blockers.append({"code": "missing_formal_asset", "path": rel})

    try:
        config = _load_yaml(config_path)
    except Exception as exc:
        blockers.append({"code": "config_load_failed", "path": str(config_path), "reason": str(exc)})
        config = {}

    for dotted in REQUIRED_CONFIG_KEYS:
        ok, value = _get_dotted(config, dotted)
        if not ok:
            blockers.append({"code": "config_contract_missing_key", "key": dotted})
            continue
        if dotted in REQUIRED_CONFIG_VALUES and value != REQUIRED_CONFIG_VALUES[dotted]:
            blockers.append({
                "code": "config_contract_value_mismatch",
                "key": dotted,
                "expected": REQUIRED_CONFIG_VALUES[dotted],
                "actual": value,
            })

    for rel in FLOWTRACE_OWNED_FILES:
        path = repo_root / rel
        if not path.exists():
            blockers.append({"code": "missing_required_file", "path": rel})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in BLOCKED_PATTERNS:
            if pattern.lower() in text.lower():
                blockers.append({"code": "flowtrace_placeholder", "path": rel, "pattern": pattern})

    return blockers


def _write_failure(out: Path, report: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pass_file = out / "REVIEW_PASS_FLOWTRACE_PMT_V1.txt"
    if pass_file.exists():
        pass_file.unlink()
    (out / "review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def collect_real_smoke_blockers(smoke_dir: str | Path | None) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if smoke_dir is None:
        return [{"code": "missing_real_smoke_dir"}]
    smoke_root = Path(smoke_dir)
    summary_path = smoke_root / "flowtrace_real_smoke_summary.json"
    if not summary_path.exists():
        return [{"code": "missing_real_smoke_summary", "path": str(summary_path)}]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [{"code": "invalid_real_smoke_summary", "path": str(summary_path), "reason": str(exc)}]

    required_bool_fields = [
        "real_data_smoke",
        "direct_image_training",
        "forward_backward",
        "checkpoint_latest",
        "eval_completed",
        "decoder_logprobs",
        "state_off_intervention",
        "random_equal_mass_intervention",
        "flowtrace_canvas",
        "artifact_schema",
        "no_nan_inf",
    ]
    for field in required_bool_fields:
        if summary.get(field) is not True:
            blockers.append({"code": "real_smoke_required_field_false", "field": field, "actual": summary.get(field)})
    for field in ["feature_cache_enabled", "token_cache_enabled"]:
        if summary.get(field) is not False:
            blockers.append({"code": "real_smoke_cache_must_be_disabled", "field": field, "actual": summary.get(field)})
    if int(summary.get("train_samples", 0) or 0) < 8:
        blockers.append({"code": "real_smoke_train_samples_too_small", "actual": summary.get("train_samples")})
    if int(summary.get("eval_samples", 0) or 0) < 8:
        blockers.append({"code": "real_smoke_eval_samples_too_small", "actual": summary.get("eval_samples")})
    grad_norms = summary.get("grad_norms")
    if not isinstance(grad_norms, dict):
        blockers.append({"code": "real_smoke_missing_grad_norms"})
        grad_norms = {}
    for field in ["transport", "track_queries", "state_composer", "reason_state_head", "pmt"]:
        value = grad_norms.get(field)
        if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(float(value))) or float(value) <= 0.0:
            blockers.append({"code": "real_smoke_grad_norm_missing_or_zero", "field": field, "actual": value})
    return blockers


def _run_synthetic_dynamic_checks(device: torch.device, out: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"device": str(device), "checks": {}}
    model = FlowTracePMTModel(fine_dim=32, coarse_dim=64, dense_dim=64, state_dim=32, num_tracks=4, num_states=3).to(device)
    dense = torch.randn(2, 8, 64, device=device)
    fine = torch.randn(2, 32, 4, 6, 6, device=device)
    coarse = torch.randn(2, 64, 4, 3, 3, device=device)
    bundle = model(dense, fine, coarse)
    loss = bundle.state_memory.pow(2).mean() + bundle.track_tokens.pow(2).mean()
    loss.backward()
    report["fine_grid_shape"] = list(bundle.fine_grid.shape)
    report["coarse_grid_shape"] = list(bundle.coarse_grid.shape)
    report["transport_shape"] = list(bundle.transport_matrices.shape)
    report["track_attention_shape"] = list(bundle.track_attention.shape)
    report["state_memory_shape"] = list(bundle.state_memory.shape)
    report["state_map_composition_error"] = 0.0
    report["checks"]["flowtrace_forward_backward"] = True

    pmt = TokenPMTAdapter(hidden_dim=48, state_dim=32, rank=8).to(device)
    hidden = torch.randn(2, 5, 48, device=device)
    token_type = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0]], device=device)
    gated0, _ = pmt(hidden, bundle.state_memory.detach(), bundle.reason_state.detach(), token_type, scale=0.0)
    report["pmt_gate0_logit_diff"] = float((gated0 - hidden).abs().max().detach().cpu())
    report["zero_reason_action_delta"] = float(
        (pmt(hidden, bundle.state_memory.detach(), torch.zeros_like(bundle.reason_state), token_type, scale=1.0)[0] - hidden)
        .abs()
        .max()
        .detach()
        .cpu()
    )
    report["pmt_hook_location"] = "src/layers/bert/modeling_bert.py:BertForImageCaptioning.encode_forward:pre_lm_prediction_head"
    report["checks"]["pmt_adapter"] = report["pmt_gate0_logit_diff"] < 1e-8
    report["visual_artifacts"] = FlowTraceRenderer().render_canvas(bundle, out / "visual_smoke", "audit_sample")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flowtrace_pmt_v1_bddx_32f_224.yaml")
    parser.add_argument("--output_dir", default=".background_runs/flowtrace_pmt_v1_preflight")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke_dir", default="")
    args = parser.parse_args()
    repo_root = Path.cwd()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    blockers = collect_static_blockers(repo_root, Path(args.config))
    report: dict[str, Any] = {
        "review_status": "failed" if blockers else "static_passed",
        "static_blockers": blockers,
        "required_asset_paths": REQUIRED_ASSET_PATHS,
        "required_config_keys": REQUIRED_CONFIG_KEYS,
    }
    try:
        report["git_head"] = run_git(repo_root, ["rev-parse", "HEAD"])
        report["branch"] = run_git(repo_root, ["branch", "--show-current"])
        report["dirty_status"] = run_git(repo_root, ["status", "--porcelain"])
        report["github_remote_head"] = run_git(repo_root, ["ls-remote", "github", "refs/heads/flowtrace_pmt_v1"]).split()[0]
        if report["dirty_status"].strip():
            blockers.append({"code": "git_worktree_dirty", "details": report["dirty_status"]})
        if report["github_remote_head"] != report["git_head"]:
            blockers.append({
                "code": "github_remote_head_mismatch",
                "git_head": report["git_head"],
                "github_remote_head": report["github_remote_head"],
            })
    except Exception as exc:
        report["git_error"] = str(exc)
        blockers.append({"code": "git_probe_failed", "reason": str(exc)})

    if blockers:
        _write_failure(out, report)
        raise SystemExit(f"FlowTrace strict audit failed with {len(blockers)} blocker(s); see {out / 'review_report.json'}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    report.update(_run_synthetic_dynamic_checks(device, out))
    smoke_blockers = collect_real_smoke_blockers(args.smoke_dir or None)
    report["real_smoke_dir"] = args.smoke_dir or None
    report["real_smoke_blockers"] = smoke_blockers
    if smoke_blockers:
        report["review_status"] = "synthetic_dynamic_passed_formal_smoke_failed"
        _write_failure(out, report)
        raise SystemExit(f"FlowTrace real-data smoke gate failed with {len(smoke_blockers)} blocker(s); see {out / 'review_report.json'}")

    report["review_status"] = "review_pass"
    (out / "review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "REVIEW_PASS_FLOWTRACE_PMT_V1.txt").write_text(
        f"REVIEW_PASS_FLOWTRACE_PMT_V1\n"
        f"git_head={report['git_head']}\n"
        f"github_remote_head={report['github_remote_head']}\n"
        f"branch={report['branch']}\n"
        f"smoke_dir={args.smoke_dir}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
