import json
from pathlib import Path

import torch

from fate_x.engine import train_acpr_flowcal_pp


def test_epoch_eval_summary_updates_best_only_on_improvement(tmp_path):
    first = {"metric_name": "test_neg_total_loss", "metric_value": -5.0, "total_loss": 5.0}
    second = {"metric_name": "test_neg_total_loss", "metric_value": -7.0, "total_loss": 7.0}
    third = {"metric_name": "test_neg_total_loss", "metric_value": -4.0, "total_loss": 4.0}

    best = None
    best, improved = train_acpr_flowcal_pp.write_epoch_eval_artifacts(tmp_path, 0, first, best)
    assert improved is True
    assert best == -5.0
    assert json.loads((tmp_path / "eval_epoch_000.json").read_text())["is_best"] is True

    best, improved = train_acpr_flowcal_pp.write_epoch_eval_artifacts(tmp_path, 1, second, best)
    assert improved is False
    assert best == -5.0
    assert json.loads((tmp_path / "eval_epoch_001.json").read_text())["is_best"] is False

    best, improved = train_acpr_flowcal_pp.write_epoch_eval_artifacts(tmp_path, 2, third, best)
    assert improved is True
    assert best == -4.0
    assert json.loads((tmp_path / "eval_epoch_002.json").read_text())["is_best"] is True

    rows = [json.loads(line) for line in (tmp_path / "eval_summary.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [0, 1, 2]
    assert [row["is_best"] for row in rows] == [True, False, True]


def test_checkpoint_best_test_is_not_overwritten_when_eval_does_not_improve(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    payload_kwargs = {
        "model": model,
        "optimizer": opt,
        "epoch_idx": 0,
        "global_step": 10,
        "optimizer_step": 2,
        "gradient_accumulation_steps": 4,
    }

    train_acpr_flowcal_pp.save_training_checkpoint(tmp_path, update_best=True, eval_report={"metric_value": 1.0}, **payload_kwargs)
    first_best_mtime = (tmp_path / "checkpoint_best_test.pth").stat().st_mtime_ns
    train_acpr_flowcal_pp.save_training_checkpoint(tmp_path, update_best=False, eval_report={"metric_value": 0.5}, **payload_kwargs)
    second_best_mtime = (tmp_path / "checkpoint_best_test.pth").stat().st_mtime_ns

    assert (tmp_path / "checkpoint_latest.pth").exists()
    assert first_best_mtime == second_best_mtime
    latest = torch.load(tmp_path / "checkpoint_latest.pth", map_location="cpu")
    assert latest["eval_report"]["metric_value"] == 0.5
    assert latest["optimizer"] is not None
    best = torch.load(tmp_path / "checkpoint_best_test.pth", map_location="cpu")
    assert best["eval_report"]["metric_value"] == 1.0
    assert best["optimizer"] is None
    assert best["checkpoint_kind"] == "best_test_model_only"
    assert best["best_checkpoint_model_only"] is True


def test_resume_state_restores_counters_and_next_epoch(tmp_path):
    ckpt = tmp_path / "checkpoint_latest.pth"
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    train_acpr_flowcal_pp.save_training_checkpoint(
        tmp_path, model, opt, epoch_idx=3, global_step=123, optimizer_step=9,
        gradient_accumulation_steps=16, update_best=True, eval_report={"metric_value": -1.0}, epoch_complete=True
    )

    new_model = torch.nn.Linear(2, 2)
    new_opt = torch.optim.AdamW(new_model.parameters())
    state = train_acpr_flowcal_pp.load_resume_state(ckpt, new_model, new_opt, device="cpu")

    assert state["start_epoch"] == 4
    assert state["optimizer_loaded"] is True


def test_model_only_best_checkpoint_can_load_without_optimizer_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    train_acpr_flowcal_pp.save_training_checkpoint(
        tmp_path,
        model,
        opt,
        epoch_idx=1,
        global_step=20,
        optimizer_step=5,
        gradient_accumulation_steps=4,
        update_best=True,
        eval_report={"metric_value": 1.0},
        epoch_complete=True,
    )

    new_model = torch.nn.Linear(2, 2)
    new_opt = torch.optim.AdamW(new_model.parameters())
    state = train_acpr_flowcal_pp.load_resume_state(tmp_path / "checkpoint_best_test.pth", new_model, new_opt, device="cpu")

    assert state["start_epoch"] == 2
    assert state["optimizer_loaded"] is False


def test_legacy_resume_without_epoch_infers_start_epoch_from_global_step(tmp_path):
    ckpt = tmp_path / "checkpoint_latest.pth"
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "global_step": 8196,
            "optimizer_step": 514,
            "gradient_accumulation_steps": 16,
        },
        ckpt,
    )

    new_model = torch.nn.Linear(2, 2)
    new_opt = torch.optim.AdamW(new_model.parameters())
    state = train_acpr_flowcal_pp.load_resume_state(ckpt, new_model, new_opt, device="cpu")

    assert state["start_epoch"] == 0
    assert train_acpr_flowcal_pp.resolve_resume_start_epoch(state, train_loader_len=8196) == 1
    assert train_acpr_flowcal_pp.resolve_resume_start_epoch(state, train_loader_len=4098) == 2


def test_resume_start_epoch_never_goes_backwards_when_epoch_field_is_stale():
    state = {
        "start_epoch": 0,
        "global_step": 8196,
        "has_epoch": True,
    }

    assert train_acpr_flowcal_pp.resolve_resume_start_epoch(state, train_loader_len=4098) == 2



def test_save_training_checkpoint_writes_separate_best_metric_files(tmp_path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    kwargs = {
        "model": model,
        "optimizer": opt,
        "epoch_idx": 2,
        "global_step": 30,
        "optimizer_step": 3,
        "gradient_accumulation_steps": 4,
        "eval_report": {"metric_value": 2.0},
    }

    train_acpr_flowcal_pp.save_training_checkpoint(
        tmp_path,
        update_best=True,
        best_checkpoint_names=["checkpoint_best_text.pth", "checkpoint_best_adapt_joint.pth"],
        **kwargs,
    )

    assert (tmp_path / "checkpoint_latest.pth").exists()
    assert (tmp_path / "checkpoint_best_text.pth").exists()
    assert (tmp_path / "checkpoint_best_adapt_joint.pth").exists()
    assert not (tmp_path / "checkpoint_best_control.pth").exists()
    best_text = torch.load(tmp_path / "checkpoint_best_text.pth", map_location="cpu")
    assert best_text["optimizer"] is None
    assert best_text["checkpoint_kind"] == "best_text_model_only"


def test_adapt_selection_scores_use_text_and_continuous_control_only():
    report = {
        "metric_value": 2.0,
        "control_metrics": {
            "available": True,
            "signals": {
                "speed": {"rmse": 2.0, "acc_at_0.5": 0.25, "acc_at_1": 0.5},
                "course": {"rmse": 4.0, "acc_at_0.5": 0.75, "acc_at_1": 1.0},
            },
            "speed_decision": {"macro_recall": 0.0},
        },
    }

    scores = train_acpr_flowcal_pp.compute_adapt_selection_scores(report)

    assert scores["text_cider"] == 2.0
    assert scores["control_rmse_negative"] == -6.0
    assert scores["control_threshold_mean"] == 0.625
    assert scores["adapt_joint"] > 0.0
    assert "speed_decision" not in json.dumps(scores)


def test_traffic_flow_audit_reports_correlations_and_top_factors():
    flow = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]], dtype=torch.float32)
    pred = torch.tensor([[[0.0, 1.0], [0.0, 0.5]], [[0.0, 1.0], [0.0, 1.5]], [[0.0, 1.0], [0.0, 0.25]]])
    target = torch.tensor([[[0.0, 1.0], [0.0, 0.4]], [[0.0, 1.0], [0.0, 1.4]], [[0.0, 1.0], [0.0, 0.2]]])

    audit = train_acpr_flowcal_pp.summarize_traffic_flow_audit(
        flow_probs=flow,
        predicate_probs=None,
        pred_control=pred,
        target_control=target,
        signal_names=["course", "speed"],
        flow_factor_names=["clear_open_flow", "queue_congestion"],
        predicate_names=[],
    )

    assert audit["sample_count"] == 3
    assert audit["flow_factors"]["clear_open_flow"]["mean"] > 0.0
    assert audit["top_flow_factors"][0]["name"] in {"clear_open_flow", "queue_congestion"}
    assert "target_speed_delta_corr" in audit["flow_factors"]["clear_open_flow"]
