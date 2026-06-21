from fate_x.utils.acpr_flow_config import load_acpr_flow_config
from fate_x.utils.acpr_flow_git_guard import _windows_gitdir_to_wsl
from fate_x.engine import train_acpr_flowcal_pp


def test_config_contract_for_direct_image_no_cache_no_legacy_and_full_plan_keys():
    cfg = load_acpr_flow_config("configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml")
    assert cfg["data"]["direct_image_training"] is True
    assert cfg["data"]["feature_cache_enabled"] is False
    assert cfg["data"]["token_cache_enabled"] is False
    assert cfg["data"]["build_cache_before_training"] is False
    assert cfg["data"]["max_num_frames"] == 32
    assert cfg["data"]["image_resolution"] == 224
    assert cfg["paths"]["adapt_checkpoint"] == "checkpoints/basemodel/checkpoints/model.bin"
    assert cfg["paths"]["bert_dir"] == "models/captioning/bert-base-uncased"
    assert cfg["paths"]["video_swin_checkpoint"] == "models/video_swin_transformer/swin_base_patch244_window877_kinetics600_22k.pth"
    assert cfg["model"]["transport"]["local_candidates"] == 25
    assert cfg["model"]["transport"]["dustbin"] is True
    assert cfg["model"]["reason_memory"]["local_tokens"] == 32
    assert cfg["model"]["reason_memory"]["flow_tokens"] == 13
    assert cfg["model"]["reason_memory"]["null_tokens"] == 1
    assert cfg["model"]["hardpair"]["pair_budget_ratio"] == 0.08
    assert cfg["model"]["seca"]["action_evidence_grad_scale"] == 0.25
    assert cfg["model"]["seca"]["explanation_evidence_grad_scale"] == 1.0
    assert cfg["model"]["control_adapter"]["direct_flow_bypass"] is False
    assert cfg["loss"]["reason_semantic"] == 0.05
    assert cfg["loss"]["future_control"] == 0.02
    assert cfg["optimization"]["precision"] == "bf16"
    assert cfg["memory_probe"]["hard_peak_reserved_limit_gib"] == 43
    assert cfg["experiment_suite"]["common_stage_a"]["epochs_0_1"]["freeze"] == ["video_swin", "bert", "sensor_head"]
    assert cfg["experiment_suite"]["fork_b2_acpr_flowcal_pp"]["transport_enabled"] is True
    assert cfg["sequence_calalign"]["fit_uses_test"] is False
    assert cfg["evaluation"]["eval_splits"] == ["test"]
    # ADAPT sep-cap evaluation uses the default greedy beam setting; beam>1 ends at the first caption SEP in this codebase.
    assert cfg["evaluation"]["beam_size_formal"] == 1
    assert cfg["evaluation"]["metric_based_early_stop"] is False
    assert cfg["supervisor"]["require_review_pass"] is True


def test_build_model_config_carries_yaml_loss_weights_into_model_config():
    cfg = load_acpr_flow_config("configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml")
    cfg["loss"] = dict(cfg["loss"])
    cfg["loss"]["control"] = 0.123
    cfg["loss"]["future_control"] = 0.045

    model_cfg = train_acpr_flowcal_pp.build_model_config(cfg, load_pretrained_backbone=False)

    assert model_cfg.loss_weights["action_text"] == 1.0
    assert model_cfg.loss_weights["explanation_text"] == 1.0
    assert model_cfg.loss_weights["control"] == 0.123
    assert model_cfg.loss_weights["predicate_pu"] == 0.05
    assert model_cfg.loss_weights["flow_pu"] == 0.03
    assert model_cfg.loss_weights["reason_semantic"] == 0.05
    assert model_cfg.loss_weights["future_control"] == 0.045
    assert model_cfg.loss_weights["memory_diversity"] == 0.001


def test_git_guard_converts_windows_worktree_gitdir_to_wsl_path():
    assert _windows_gitdir_to_wsl("E:/sbw/ADAPT_repro/ADAPT/.git/worktrees/x") == "/mnt/e/sbw/ADAPT_repro/ADAPT/.git/worktrees/x"
    assert _windows_gitdir_to_wsl("C:\\Users\\WLJTXY\\repo\\.git") == "/mnt/c/Users/WLJTXY/repo/.git"
