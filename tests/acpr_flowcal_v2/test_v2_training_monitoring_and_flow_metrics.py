from __future__ import annotations

import json
import inspect
from types import SimpleNamespace

import h5py
import pytest
import torch

from fate_x.engine import eval_acpr_flowcal_v2 as eval_mod
from fate_x.engine import acpr_flowcal_v2_data as data_mod
from fate_x.engine.adapt_caption_eval_bridge import caption_file_from_loader, write_adapt_sep_caption_tsv
from fate_x.engine import train_acpr_flowcal_v2 as train_mod


class _TinyBatch:
    def __init__(self, sample_id: str = "sample0"):
        self.frames = torch.zeros(1, 4, 3, 16, 16)
        self.input_ids = None
        self.attention_mask = None
        self.token_type_ids = None
        self.masked_pos = None
        self.masked_ids = None
        self.car_info = torch.zeros(1, 2, 4)
        self.sample_ids = [sample_id]


class _TinyTrainLoader:
    def __len__(self):
        return 4

    def __iter__(self):
        for idx in range(4):
            yield _TinyBatch(f"train{idx}")


class _LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch, stage="R"):
        loss = self.weight * 1.0
        zero = loss * 0.0
        return type(
            "Out",
            (),
            {
                "total_loss": loss,
                "action_text_loss": loss,
                "speed_loss": zero,
                "course_loss": zero,
                "auxiliary_loss": zero,
                "loss_components": {"text": loss, "speed": zero, "course": zero, "auxiliary": zero},
            },
        )()


class _CountingOptimizer:
    def __init__(self, params):
        self.inner = torch.optim.SGD(params, lr=0.01)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.inner.step()

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)


class _CountingScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


def test_train_one_epoch_honors_gradient_accumulation_steps(tmp_path):
    model = _LossModel()
    optimizer = _CountingOptimizer(model.parameters())
    scheduler = _CountingScheduler()

    metrics = train_mod.train_one_epoch(
        model,
        _TinyTrainLoader(),
        optimizer,
        scheduler,
        epoch=0,
        device="cpu",
        gradient_accumulation_steps=2,
        log_interval=1,
        log_path=tmp_path / "train_progress.jsonl",
    )

    assert metrics["steps"] == 4.0
    assert metrics["optimizer_steps"] == 2.0
    assert optimizer.step_count == 2
    assert scheduler.step_count == 2
    rows = (tmp_path / "train_progress.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 4


def test_stage_weighted_loss_honors_config_and_control_gate():
    cfg = train_mod.FlowCalV2Config.from_dict(
        {
            "loss": {
                "action_text": 1.0,
                "explanation_text": 1.0,
                "speed_normalized": 0.03,
                "course_normalized": 0.03,
            },
            "stages": [
                {"name": "semantic_recovery", "epochs": [0], "enable_control_losses": False},
                {"name": "axis_aware_motion", "epochs": [1], "enable_control_losses": True},
            ],
            "training": {"epochs": 2},
        }
    )
    out = SimpleNamespace(
        total_loss=torch.tensor(311.0),
        action_text_loss=torch.tensor(10.0),
        speed_loss=torch.tensor(100.0),
        course_loss=torch.tensor(200.0),
        auxiliary_loss=torch.tensor(1.0),
        loss_components={
            "text": torch.tensor(10.0),
            "speed": torch.tensor(100.0),
            "course": torch.tensor(200.0),
            "auxiliary": torch.tensor(1.0),
        },
    )

    semantic_loss, semantic_weights = train_mod.compute_stage_weighted_loss(out, cfg, "semantic_recovery")
    axis_loss, axis_weights = train_mod.compute_stage_weighted_loss(out, cfg, "axis_aware_motion")

    assert torch.isclose(semantic_loss, torch.tensor(11.0))
    assert semantic_weights["speed"] == 0.0
    assert semantic_weights["course"] == 0.0
    assert semantic_weights["control_enabled"] == 0.0
    assert torch.isclose(axis_loss, torch.tensor(20.0))
    assert axis_weights["speed"] == 0.03
    assert axis_weights["course"] == 0.03
    assert axis_weights["control_enabled"] == 1.0


def test_config_parses_v2_optimizer_learning_rates():
    cfg = train_mod.FlowCalV2Config.from_dict(
        {
            "optimization": {
                "learning_rates": {
                    "transport": 5e-5,
                    "temporal_seca": 2e-5,
                    "axis_control_adapter": 2e-5,
                },
                "weight_decay": {
                    "new_modules": 0.01,
                    "backbone": 0.05,
                    "bias_norm_gate": 0.0,
                },
            },
            "training": {"epochs": 1},
            "stages": [{"name": "semantic_recovery", "epochs": [0], "enable_control_losses": False}],
        }
    )

    assert cfg.optimization_learning_rates["transport"] == 5e-5
    assert cfg.optimization_learning_rates["temporal_seca"] == 2e-5
    assert cfg.optimization_learning_rates["axis_control_adapter"] == 2e-5
    assert cfg.optimization_weight_decay["bias_norm_gate"] == 0.0


def test_optimizer_groups_use_configured_module_learning_rates():
    class _OptModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transport = torch.nn.Linear(2, 2)
            self.seca = torch.nn.Linear(2, 2)
            self.control_adapter = torch.nn.Linear(2, 2)
            self.other = torch.nn.Linear(2, 2)

    cfg = train_mod.FlowCalV2Config.from_dict(
        {
            "optimization": {
                "learning_rates": {
                    "new_modules": 7e-5,
                    "transport": 5e-5,
                    "temporal_seca": 2e-5,
                    "axis_control_adapter": 2e-5,
                }
            },
            "training": {"epochs": 1},
            "stages": [{"name": "semantic_recovery", "epochs": [0], "enable_control_losses": False}],
        }
    )
    model = _OptModel()
    groups = train_mod.build_optimizer_groups(model, cfg)
    lr_by_param = {id(param): group["lr"] for group in groups for param in group["params"]}

    assert lr_by_param[id(model.transport.weight)] == 5e-5
    assert lr_by_param[id(model.seca.weight)] == 2e-5
    assert lr_by_param[id(model.control_adapter.weight)] == 2e-5
    assert lr_by_param[id(model.other.weight)] == 7e-5


def test_epoch_safety_rejects_text_collapse_and_missing_prediction_flow_corr():
    safe = train_mod.assess_epoch_safety(
        {
            "text_metrics_available": True,
            "CIDEr_des": 0.16,
            "CIDEr_exp": 0.04,
            "pred_speed_delta_corr": 0.02,
            "pred_course_delta_corr": -0.01,
        },
        best_safe_text_sum=0.20,
    )
    assert safe["accepted"] is True

    collapsed = train_mod.assess_epoch_safety(
        {
            "text_metrics_available": True,
            "CIDEr_des": 0.001,
            "CIDEr_exp": 0.001,
            "pred_speed_delta_corr": None,
            "pred_course_delta_corr": 0.01,
        },
        best_safe_text_sum=0.20,
    )
    assert collapsed["accepted"] is False
    assert "adapt_text_metric_negative_migration" in collapsed["reasons"]
    assert "adapt_text_metric_collapse" in collapsed["reasons"]
    assert "missing_pred_speed_delta_corr" in collapsed["reasons"]


def test_epoch_safety_rejects_resume_text_negative_migration():
    safety = train_mod.assess_epoch_safety(
        {
            "text_metrics_available": True,
            "CIDEr_des": 0.07819261921358316,
            "CIDEr_exp": 0.02855993590466849,
            "pred_speed_delta_corr": -0.022295216098427773,
            "pred_course_delta_corr": -0.020563744008541107,
        },
        best_safe_text_sum=0.1911726210069978,
    )

    assert safety["accepted"] is False
    assert safety["text_sum"] == pytest.approx(0.10675255511825164)
    assert "adapt_text_metric_negative_migration" in safety["reasons"]


def test_best_checkpoint_suite_seed_prevents_lower_resume_overwrite(tmp_path):
    resume_metrics = {
        "text_metrics_available": True,
        "CIDEr_des": 0.1717788683892075,
        "CIDEr_exp": 0.019393752617790305,
        "METEOR_exp": 0.02,
        "speed_rmse": 6.924437999725342,
        "course_rmse": 89.10369110107422,
        "speed_acc@1": 0.20,
        "course_acc@1": 0.003,
    }
    lower_metrics = {
        "text_metrics_available": True,
        "CIDEr_des": 0.07819261921358316,
        "CIDEr_exp": 0.02855993590466849,
        "METEOR_exp": 0.027,
        "speed_rmse": 6.924335479736328,
        "course_rmse": 89.10367584228516,
        "speed_acc@1": 0.21,
        "course_acc@1": 0.003,
    }
    suite = train_mod.BestCheckpointSuite(initial_metrics=resume_metrics)

    updates = suite.update_and_save(tmp_path, {"model": {}, "epoch": 4}, lower_metrics)

    assert "checkpoint_best_text.pth" not in updates
    assert "checkpoint_best_joint.pth" not in updates
    assert "checkpoint_best_control.pth" in updates


def test_negative_migration_safety_runs_before_best_checkpoint_update(tmp_path):
    resume_metrics = {
        "text_metrics_available": True,
        "CIDEr_des": 0.1717788683892075,
        "CIDEr_exp": 0.019393752617790305,
        "METEOR_exp": 0.02,
        "speed_rmse": 6.924437999725342,
        "course_rmse": 89.10369110107422,
        "speed_acc@1": 0.20,
        "course_acc@1": 0.003,
    }
    epoch4_metrics = {
        "text_metrics_available": True,
        "CIDEr_des": 0.07819261921358316,
        "CIDEr_exp": 0.02855993590466849,
        "METEOR_exp": 0.027,
        "speed_rmse": 6.924335479736328,
        "course_rmse": 89.10367584228516,
        "speed_acc@1": 0.21,
        "course_acc@1": 0.003,
        "pred_speed_delta_corr": -0.022295216098427773,
        "pred_course_delta_corr": -0.020563744008541107,
    }
    suite = train_mod.BestCheckpointSuite(initial_metrics=resume_metrics)
    safety = train_mod.assess_epoch_safety(
        epoch4_metrics,
        best_safe_text_sum=train_mod._text_sum_value(resume_metrics),
    )

    assert safety["accepted"] is False
    assert "adapt_text_metric_negative_migration" in safety["reasons"]
    if not safety["accepted"]:
        updates = []
    else:
        updates = suite.update_and_save(tmp_path, {"model": {}, "epoch": 4}, epoch4_metrics)

    assert updates == []


def test_eval_failure_record_requires_epoch_checkpoint_before_resume(tmp_path):
    checkpoint = tmp_path / "checkpoint_latest.pth"
    checkpoint.write_bytes(b"checkpoint")
    err = RuntimeError("caption eval crashed")

    record = train_mod.write_eval_failure_record(
        tmp_path,
        epoch=1,
        checkpoint_path=checkpoint,
        eval_output_dir=tmp_path / "eval_epoch_001",
        exc=err,
    )

    assert record["epoch"] == 1
    assert record["checkpoint_path"] == str(checkpoint)
    assert record["resume_forbidden_until_epoch_eval"] is True
    rows = (tmp_path / "eval_failures.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    saved = json.loads(rows[0])
    assert saved["action_required"].startswith("fix evaluation logic")


def test_pending_eval_recovery_evaluates_failed_epoch_checkpoint_before_training(monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1)
    checkpoint = tmp_path / "checkpoint_latest.pth"
    torch.save({"model": model.state_dict(), "epoch": 1}, checkpoint)
    eval_dir = tmp_path / "eval_epoch_001"
    run_manifest = {
        "eval_failed": True,
        "pending_eval_epoch": 1,
        "pending_eval_checkpoint": str(checkpoint),
        "pending_eval_output_dir": str(eval_dir),
        "best_safe_text_sum": 0.10,
    }
    calls = []

    def fake_evaluate_after_epoch(received_model, loader, epoch, device="cpu", output_dir=None):
        calls.append({"epoch": epoch, "output_dir": str(output_dir)})
        return {
            "text_metrics_available": True,
            "CIDEr_des": 0.08,
            "CIDEr_exp": 0.04,
            "speed_rmse": 1.0,
            "course_rmse": 2.0,
            "pred_speed_delta_corr": 0.1,
            "pred_course_delta_corr": 0.2,
        }

    monkeypatch.setattr(train_mod, "evaluate_after_epoch", fake_evaluate_after_epoch)

    recovered = train_mod.recover_pending_eval_before_training(
        tmp_path,
        model=model,
        test_loader=[],
        device="cpu",
        run_manifest=run_manifest,
        checkpoint_suite=train_mod.BestCheckpointSuite(),
        best_safe_text_sum=0.10,
    )

    assert recovered["recovered"] is True
    assert calls == [{"epoch": 1, "output_dir": str(eval_dir)}]
    rows = (tmp_path / "metrics_summary.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    saved_metrics = json.loads(rows[0])
    assert saved_metrics["epoch"] == 1
    assert saved_metrics["recovered_pending_eval"] is True
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["eval_failed"] is False
    assert manifest["pending_eval_resolved"] is True
    assert manifest["last_completed_epoch"] == 1


def test_formal_suite_recovers_pending_eval_and_does_not_train_next_epoch(monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1)
    checkpoint = tmp_path / "checkpoint_latest.pth"
    torch.save({"model": model.state_dict(), "epoch": 1}, checkpoint)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "eval_failed": True,
                "pending_eval_epoch": 1,
                "pending_eval_checkpoint": str(checkpoint),
                "pending_eval_output_dir": str(tmp_path / "eval_epoch_001"),
                "best_safe_text_sum": 0.10,
            }
        ),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(epochs=3, optimization_learning_rates={}, optimization_weight_decay={})
    monkeypatch.setattr(train_mod, "load_flowcal_v2_config", lambda path: cfg)
    monkeypatch.setattr(train_mod, "ACPRFlowCalV2Model", lambda config, **kwargs: model)
    monkeypatch.setattr(train_mod, "build_v2_dataloader", lambda split, **kwargs: _TinyTrainLoader())

    def fail_if_train_called(*args, **kwargs):
        raise AssertionError("training must not continue until pending epoch eval is recovered")

    monkeypatch.setattr(train_mod, "train_one_epoch", fail_if_train_called)
    monkeypatch.setattr(
        train_mod,
        "evaluate_after_epoch",
        lambda received_model, loader, epoch, device="cpu", output_dir=None: {
            "text_metrics_available": True,
            "CIDEr_des": 0.08,
            "CIDEr_exp": 0.04,
            "speed_rmse": 1.0,
            "course_rmse": 2.0,
            "pred_speed_delta_corr": 0.1,
            "pred_course_delta_corr": 0.2,
        },
    )

    result = train_mod.run_formal_suite(
        "config.yaml",
        str(tmp_path),
        device="cpu",
        epochs=3,
        batch_size=1,
        num_workers=0,
        synthetic_smoke=True,
    )

    assert result["pending_eval_recovered"] is True
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["eval_failed"] is False
    assert manifest["pending_eval_resolved"] is True
    assert manifest["last_completed_epoch"] == 1


def test_rejected_epoch_record_keeps_bad_epoch_out_of_best_contract(tmp_path):
    metrics = {
        "text_metrics_available": True,
        "CIDEr_des": 0.001,
        "CIDEr_exp": 0.001,
        "speed_rmse": 1.0,
        "course_rmse": 2.0,
        "pred_speed_delta_corr": 0.1,
        "pred_course_delta_corr": 0.1,
    }
    safety = train_mod.assess_epoch_safety(metrics, best_safe_text_sum=0.20)
    assert safety["accepted"] is False

    record = train_mod.write_epoch_rejection_record(
        tmp_path,
        epoch=4,
        checkpoint_path=tmp_path / "checkpoint_rejected_epoch_004.pth",
        metrics=metrics,
        safety=safety,
    )

    assert record["action_required"].startswith("resume from the last safe")
    rows = (tmp_path / "safety_rejections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert "adapt_text_metric_collapse" in json.loads(rows[0])["safety"]["reasons"]


def test_train_progress_logs_weighted_and_raw_losses(tmp_path):
    cfg = train_mod.FlowCalV2Config.from_dict({"stages": [{"name": "semantic_recovery", "epochs": [0], "enable_control_losses": False}], "training": {"epochs": 1}})
    model = _LossModel()
    optimizer = _CountingOptimizer(model.parameters())
    scheduler = _CountingScheduler()

    train_mod.train_one_epoch(model, _TinyTrainLoader(), optimizer, scheduler, epoch=0, device="cpu", gradient_accumulation_steps=2, log_interval=1, log_path=tmp_path / "train_progress.jsonl", config=cfg, stage="semantic_recovery")

    row = json.loads((tmp_path / "train_progress.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "weighted_loss" in row
    assert "raw_total_loss" in row
    assert row["loss_weights"]["control_enabled"] == 0.0


def test_formal_suite_exposes_exact_resume_contract(tmp_path):
    signature = inspect.signature(train_mod.run_formal_suite)
    assert "resume_checkpoint" in signature.parameters

    model = _LossModel()
    tmp_resume = tmp_path / "checkpoint_latest.pth.tmp"
    torch.save({"model": model.state_dict(), "epoch": 0}, tmp_resume)
    try:
        train_mod.load_resume_exact(tmp_resume, model)
    except ValueError as exc:
        assert ".tmp resume is forbidden" in str(exc)
    else:
        raise AssertionError(".tmp resume path should be rejected")


def test_resume_skips_incompatible_optimizer_state_after_group_layout_change(tmp_path):
    class _TwoParamModel(_LossModel):
        def __init__(self):
            super().__init__()
            self.extra = torch.nn.Parameter(torch.tensor(2.0))

    model = _TwoParamModel()
    checkpoint = tmp_path / "checkpoint_best_joint.pth"
    old_optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 1e-4}])
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": old_optimizer.state_dict(),
            "scheduler": {"step_count": 5, "base_lrs": [1e-4]},
            "epoch": 3,
        },
        checkpoint,
    )
    new_optimizer = torch.optim.AdamW(
        [
            {"params": [model.weight], "lr": 5e-5},
            {"params": [model.extra], "lr": 2e-5},
        ]
    )
    scheduler = train_mod.StageAwareScheduler(new_optimizer, total_steps=10)

    payload = train_mod.load_resume_exact(checkpoint, model, optimizer=new_optimizer, scheduler=scheduler)

    assert payload["epoch"] == 3
    assert payload["resume_meta"]["optimizer_resume_loaded"] is False
    assert payload["resume_meta"]["scheduler_resume_loaded"] is False
    assert payload["resume_meta"]["scheduler_progress_fast_forwarded"] is True
    assert payload["resume_meta"]["scheduler_progress_step_count"] == 5
    assert scheduler.step_count == 5


class _EvalModel(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, batch, stage="S"):
        target = batch.car_info.transpose(1, 2)
        pred = target * 0.8
        reason_value = target[..., 1].mean()
        bundle = type(
            "Bundle",
            (),
            {
                "diagnostics": {
                    "traffic_density": torch.tensor(1.0),
                    "traffic_density_per_sample": torch.ones(target.shape[0]),
                    "traffic_motion_per_sample": target[..., 1].mean(dim=1),
                    "transport_dustbin": torch.tensor(0.1),
                }
            },
        )()
        return type(
            "Out",
            (),
            {
                "total_loss": torch.tensor(1.0),
                "action_text_loss": torch.tensor(0.0),
                "explanation_text_loss": torch.tensor(0.0),
                "control_final_prediction": pred,
                "bundle": bundle,
            },
        )()

    def generate_adapt_caption_pairs(self, batch, tokenizer=None):
        return [{"img_key": batch.sample_ids[0], "description": "go forward", "explanation": "because traffic is clear"}]


def test_eval_reports_prediction_and_target_flow_delta_correlations(monkeypatch, tmp_path):
    batches = []
    for idx, base in enumerate((1.0, 2.0, 4.0)):
        batch = _TinyBatch(f"eval{idx}")
        speed = torch.tensor([0.0, base, base * 2.0, base * 3.0])
        course = torch.tensor([0.0, base, base * 2.0, base * 3.0])
        batch.car_info = torch.stack([course, speed], dim=0).unsqueeze(0)
        batches.append(batch)

    monkeypatch.setattr(eval_mod, "_move_batch_to_device", lambda batch, device: batch)
    monkeypatch.setattr(
        eval_mod,
        "run_adapt_sep_caption_eval",
        lambda rows, loader, output_dir: {
            "text_metrics_available": True,
            "CIDEr_des": 0.1,
            "CIDEr_exp": 0.2,
        },
    )

    metrics = eval_mod.evaluate(_EvalModel(), batches, device="cpu", output_dir=tmp_path)

    assert metrics["target_speed_delta_corr"] is not None
    assert metrics["target_course_delta_corr"] is not None
    assert metrics["pred_speed_delta_corr"] is not None
    assert metrics["pred_course_delta_corr"] is not None
    assert "traffic_flow_audit" in metrics
    audit = metrics["traffic_flow_audit"]
    assert audit["traffic_factor_std"] > 0
    assert audit["primary_factor"] == "traffic_motion"
    assert "traffic_motion" in audit["factors"]
    assert audit["factors"]["traffic_density"]["std"] == 0.0
    assert audit["factors"]["traffic_motion"]["std"] > 0
    assert audit["target_speed_delta_std"] > 0
    assert audit["target_course_delta_std"] > 0
    assert audit["pred_speed_delta_std"] > 0
    assert audit["pred_course_delta_std"] > 0


def test_formal_dataloader_default_does_not_limit_eval_samples(monkeypatch):
    seen = {}

    def fake_build_args(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(), object()

    class _FakeLoader:
        dataset = object()

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 123

    monkeypatch.setattr(data_mod, "_build_adapt_args", fake_build_args)
    monkeypatch.setattr("src.datasets.vl_dataloader.make_data_loader", lambda *args, **kwargs: _FakeLoader())

    loader = data_mod.build_v2_dataloader("test", batch_size=4, num_workers=0, formal=True)

    assert len(loader) == 123
    assert seen["limited_samples"] == -1


def test_flowcal_v2_loader_adapter_exposes_underlying_caption_dataset():
    class _CaptionDataset:
        caption_file = "fallback_caption.json"

        def get_caption_file_in_coco_format(self):
            return "official_caption.json"

    class _FakeLoader:
        dataset = _CaptionDataset()

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    adapter = data_mod._FlowCalV2LoaderAdapter(_FakeLoader())

    assert adapter.dataset is _FakeLoader.dataset
    assert caption_file_from_loader(adapter) == "official_caption.json"


def test_flowcal_v2_loader_adapter_exposes_tokenizer_for_caption_decode():
    tokenizer = object()

    class _FakeLoader:
        dataset = object()

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    adapter = data_mod._FlowCalV2LoaderAdapter(_FakeLoader(), tokenizer=tokenizer)

    assert adapter.tokenizer is tokenizer


def test_evaluate_after_epoch_passes_loader_tokenizer(monkeypatch, tmp_path):
    tokenizer = object()
    loader = SimpleNamespace(tokenizer=tokenizer)
    captured = {}

    def fake_evaluate(model, loader_arg, **kwargs):
        captured.update(kwargs)
        return {"text_metrics_available": True}

    monkeypatch.setattr(train_mod, "evaluate_v2_test", fake_evaluate)

    metrics = train_mod.evaluate_after_epoch(_LossModel(), loader, epoch=1, device="cpu", output_dir=tmp_path)

    assert captured["tokenizer"] is tokenizer
    assert metrics["eval_epoch"] == 1
    assert metrics["adapt_text_metric_source"] == "adapt_sep_caption_eval"


def test_adapt_caption_tsv_strips_segment_suffix_for_coco_image_ids(tmp_path):
    pred_path = write_adapt_sep_caption_tsv(
        [
            {
                "img_key": "testing_1f0fff77-a50aae97_23662:0",
                "description": "the car goes forward",
                "explanation": "because the lane is clear",
            }
        ],
        tmp_path / "pred.tsv",
    )

    first_key = pred_path.read_text(encoding="utf-8").split("\t", 1)[0]
    assert first_key == "testing_1f0fff77-a50aae97_23662"


def test_adapt_eval_batch_replaces_placeholder_car_info_from_processed_h5(monkeypatch, tmp_path):
    processed_dir = tmp_path / "processed_video_info"
    processed_dir.mkdir()
    sample_id = "testing_demo-video_12345"
    with h5py.File(processed_dir / f"{sample_id}.h5", "w") as h5:
        h5.create_dataset("course", data=torch.linspace(10.0, 41.0, 32).numpy())
        h5.create_dataset("speed", data=torch.linspace(2.0, 9.75, 32).numpy())
    monkeypatch.setenv("FATE_X_BDDX_PROCESSED_VIDEO_INFO_DIR", str(processed_dir))

    batch_size = 1
    input_ids = torch.zeros(batch_size, 30, dtype=torch.long)
    attention_mask = torch.ones(batch_size, 30, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, 30, dtype=torch.long)
    frames = torch.zeros(batch_size, 32, 3, 16, 16)
    masked_pos = torch.zeros(batch_size, 4, dtype=torch.long)
    placeholder_car_info = torch.full((batch_size, 2, 32), -1.0)
    batch = (
        [f"{sample_id}:0"],
        [input_ids, attention_mask, token_type_ids, frames, masked_pos, placeholder_car_info],
        {"sample_id": [f"{sample_id}:0"]},
    )

    adapted = data_mod.adapt_batch_to_v2(batch)

    assert adapted.car_info is not None
    assert adapted.car_info.shape == (1, 2, 32)
    assert torch.allclose(adapted.car_info[0, 0], torch.linspace(10.0, 41.0, 32))
    assert torch.allclose(adapted.car_info[0, 1], torch.linspace(2.0, 9.75, 32))
