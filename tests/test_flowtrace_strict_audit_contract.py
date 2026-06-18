from pathlib import Path

import torch

from fate_x.engine.audit_flowtrace_pmt_implementation import (
    _windows_gitdir_to_wsl,
    collect_real_smoke_blockers,
    collect_static_blockers,
)
from fate_x.engine.checkpoint_utils import filter_compatible_state_dict
from fate_x.losses.flowtrace_losses import FlowTraceLoss
from fate_x.models.flowtrace_pmt_model import FlowTracePMTModel


def test_strict_audit_flags_formal_blockers_with_incomplete_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    incomplete_config = tmp_path / "incomplete.yaml"
    incomplete_config.write_text("config_version: flowtrace_pmt_v1\n", encoding="utf-8")
    blockers = collect_static_blockers(root, incomplete_config)
    codes = {b["code"] for b in blockers}
    assert "config_contract_missing_key" in codes


def test_flowtrace_model_parameters_exist_before_forward_for_optimizer_registration():
    model = FlowTracePMTModel(fine_dim=8, coarse_dim=8, dense_dim=16, state_dim=16, num_tracks=3, num_states=2)
    names = {name for name, _ in model.named_parameters()}
    assert any(name.startswith("grid.") for name in names)
    assert any(name.startswith("tracks.") for name in names)
    assert any(name.startswith("composer.") for name in names)
    assert any(name.startswith("reason.") for name in names)


def test_flowtrace_loss_exposes_required_components():
    required = {
        "anchor",
        "reason_state",
        "dynamic_control",
        "transport_cycle",
        "track_diversity",
        "state_diversity",
        "state_sparsity",
        "preserve_kl",
        "intervention",
    }
    available = set(FlowTraceLoss.available_components())
    assert required.issubset(available)


def test_pretrained_checkpoint_loader_skips_shape_mismatched_sensor_head():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))
    checkpoint = {
        "0.weight": torch.ones(4, 2),
        "0.bias": torch.ones(3),
        "missing.weight": torch.ones(1),
    }

    filtered, skipped = filter_compatible_state_dict(model, checkpoint)

    assert set(filtered) == {"0.bias"}
    assert skipped["0.weight"]["reason"] == "shape_mismatch"
    assert skipped["missing.weight"]["reason"] == "missing_in_model"


def test_windows_gitdir_path_maps_to_wsl_mount_path():
    assert _windows_gitdir_to_wsl("E:/sbw/ADAPT_repro/ADAPT/.git/worktrees/x") == (
        "/mnt/e/sbw/ADAPT_repro/ADAPT/.git/worktrees/x"
    )
    assert _windows_gitdir_to_wsl("C:\\Users\\Name\\repo\\.git") == "/mnt/c/Users/Name/repo/.git"
    assert _windows_gitdir_to_wsl("/mnt/e/repo/.git") == "/mnt/e/repo/.git"


def test_real_smoke_gate_requires_complete_train_eval_and_artifact_evidence(tmp_path):
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()

    blockers = collect_real_smoke_blockers(smoke_dir)
    assert {b["code"] for b in blockers} == {"missing_real_smoke_summary"}

    (smoke_dir / "flowtrace_real_smoke_summary.json").write_text(
        """{
          "real_data_smoke": true,
          "train_samples": 8,
          "eval_samples": 8,
          "direct_image_training": true,
          "feature_cache_enabled": false,
          "token_cache_enabled": false,
          "forward_backward": true,
          "checkpoint_latest": true,
          "eval_completed": true,
          "decoder_logprobs": true,
          "state_off_intervention": true,
          "random_equal_mass_intervention": true,
          "flowtrace_canvas": true,
          "artifact_schema": true,
          "no_nan_inf": true,
          "grad_norms": {
            "transport": 0.1,
            "track_queries": 0.1,
            "state_composer": 0.1,
            "reason_state_head": 0.1,
            "pmt": 0.1
          }
        }""",
        encoding="utf-8",
    )

    assert collect_real_smoke_blockers(smoke_dir) == []
