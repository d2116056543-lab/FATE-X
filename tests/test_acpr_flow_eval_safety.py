from __future__ import annotations

from fate_x.engine.train_acpr_flowcal_pp import build_eval_safety_plan, resolve_control_eval_yaml


def test_eval_safety_defaults_to_pre_eval_checkpoint_and_no_smoke():
    plan = build_eval_safety_plan({})

    assert plan["save_pre_eval_checkpoint"] is True
    assert plan["smoke_before_full"] is False
    assert plan["smoke_max_eval_samples"] == 32
    assert plan["control_smoke_max_eval_samples"] == 64


def test_eval_safety_plan_reads_config_values():
    plan = build_eval_safety_plan(
        {
            "save_pre_eval_checkpoint": False,
            "smoke_before_full": True,
            "smoke_max_eval_samples": 16,
            "control_smoke_max_eval_samples": 24,
        }
    )

    assert plan == {
        "save_pre_eval_checkpoint": False,
        "smoke_before_full": True,
        "smoke_max_eval_samples": 16,
        "control_smoke_max_eval_samples": 24,
    }


def test_control_eval_uses_signal_specific_yaml_for_test_split():
    cfg = {
        "paths": {
            "train_yaml": "datasets_part/BDDX/training_32frames.yaml",
            "test_yaml": "datasets/BDDX/testing_32frames.yaml",
            "signal_test_yaml": "datasets_part/BDDX/testing_32frames.yaml",
        }
    }

    assert resolve_control_eval_yaml(cfg, "test") == "datasets_part/BDDX/testing_32frames.yaml"
    assert resolve_control_eval_yaml(cfg, "train") == "datasets_part/BDDX/training_32frames.yaml"


def test_control_eval_falls_back_to_text_yaml_when_signal_yaml_absent():
    cfg = {
        "paths": {
            "train_yaml": "datasets_part/BDDX/training_32frames.yaml",
            "test_yaml": "datasets/BDDX/testing_32frames.yaml",
        }
    }

    assert resolve_control_eval_yaml(cfg, "test") == "datasets/BDDX/testing_32frames.yaml"
