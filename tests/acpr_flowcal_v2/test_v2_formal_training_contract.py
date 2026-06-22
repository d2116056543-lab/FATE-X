from __future__ import annotations

from pathlib import Path

import torch

from fate_x.engine import train_acpr_flowcal_v2 as train_mod


class _TinyModel(torch.nn.Module):
    def __init__(self, _cfg):
        super().__init__()
        self.transport = torch.nn.Linear(1, 1)

    def forward(self, batch, stage="R"):
        loss = self.transport.weight.sum() * 0.0 + self.transport.bias.sum() * 0.0 + 1.0
        return type(
            "Out",
            (),
            {
                "total_loss": loss,
                "action_text_loss": loss * 0.0,
                "explanation_text_loss": loss * 0.0,
                "control_final_prediction": torch.zeros(1, 2, 2),
                "bundle": type("Bundle", (), {"diagnostics": {}})(),
            },
        )()


class _TinyBatch:
    def __init__(self):
        self.frames = torch.zeros(1, 4, 3, 16, 16)
        self.input_ids = None
        self.masked_pos = None
        self.masked_ids = None
        self.car_info = torch.zeros(1, 2, 4)
        self.sample_ids = ["sample0"]


class _TinyLoader:
    def __init__(self):
        self.dataset = type("Dataset", (), {"yaml_file": "BDDX/testing_32frames.yaml"})()

    def __len__(self):
        return 1

    def __iter__(self):
        yield _TinyBatch()


def test_formal_suite_does_not_default_to_synthetic_length_two(monkeypatch, tmp_path):
    calls = []

    def fake_loader(split, **kwargs):
        calls.append((split, kwargs))
        return _TinyLoader()

    monkeypatch.setattr(train_mod, "build_v2_dataloader", fake_loader)
    monkeypatch.setattr(train_mod, "ACPRFlowCalV2Model", _TinyModel)
    monkeypatch.setattr(
        train_mod,
        "evaluate_after_epoch",
        lambda *args, **kwargs: {
            "CIDEr_des": 1.0,
            "CIDEr_exp": 0.5,
            "METEOR_exp": 0.1,
            "speed_rmse": 1.0,
            "course_rmse": 1.0,
            "speed_acc@1": 0.2,
            "course_acc@1": 0.2,
            "text_metrics_available": True,
        },
    )

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
data:
  direct_image_training: true
  feature_cache_enabled: false
  token_cache_enabled: false
  max_num_frames: 4
model:
  state_dim: 16
  text_hidden_dim: 32
  adapt_feature_dim: 512
training:
  epochs: 1
stages:
  - name: semantic_recovery
    epochs: [0]
""",
        encoding="utf-8",
    )
    train_mod.run_formal_suite(str(cfg), str(tmp_path / "run"), device="cpu", epochs=1, batch_size=2, num_workers=3)

    assert calls
    assert calls[0][1].get("formal") is True
    assert calls[0][1].get("synthetic") is False
    assert calls[0][1].get("config_path") == str(cfg)
    assert "length" not in calls[0][1]
    assert "length" not in calls[1][1]


def test_best_checkpoint_suite_writes_all_required_files(tmp_path):
    suite = train_mod.BestCheckpointSuite()
    payload = {"model": {"w": torch.tensor(1.0)}, "epoch": 0}
    metrics = {
        "CIDEr_des": 1.0,
        "CIDEr_exp": 2.0,
        "METEOR_exp": 0.4,
        "speed_rmse": 0.5,
        "course_rmse": 0.25,
        "speed_acc@1": 0.8,
        "course_acc@1": 0.75,
        "text_metrics_available": True,
    }

    updated = suite.update_and_save(tmp_path, payload, metrics)

    assert set(updated) == {
        "checkpoint_best_text.pth",
        "checkpoint_best_explanation.pth",
        "checkpoint_best_control.pth",
        "checkpoint_best_joint.pth",
        "checkpoint_best_test.pth",
    }
    for name in updated:
        assert (tmp_path / name).exists()


def test_text_best_is_not_updated_without_adapt_text_metrics(tmp_path):
    suite = train_mod.BestCheckpointSuite()
    payload = {"model": {"w": torch.tensor(1.0)}, "epoch": 0}
    metrics = {
        "speed_rmse": 0.5,
        "course_rmse": 0.25,
        "speed_acc@1": 0.8,
        "course_acc@1": 0.75,
        "text_metrics_available": False,
    }

    updated = suite.update_and_save(tmp_path, payload, metrics)

    assert "checkpoint_best_control.pth" in updated
    assert "checkpoint_best_text.pth" not in updated
    assert "checkpoint_best_explanation.pth" not in updated
    assert "checkpoint_best_joint.pth" not in updated
