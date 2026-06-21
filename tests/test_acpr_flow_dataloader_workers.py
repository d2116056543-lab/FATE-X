from __future__ import annotations

from fate_x.engine import acpr_bddx_data


def _base_cfg(allow_windows_workers: bool):
    return {
        "paths": {"bert_dir": "models/captioning/bert-base-uncased"},
        "data": {
            "image_resolution": 224,
            "max_num_frames": 32,
            "num_workers": 4,
            "persistent_workers": True,
            "allow_windows_workers": allow_windows_workers,
        },
    }


def test_windows_workers_default_to_safe_zero(monkeypatch):
    monkeypatch.setattr(acpr_bddx_data.os, "name", "nt", raising=False)

    args = acpr_bddx_data.build_bddx_acpr_args(_base_cfg(False), split="train", batch_size=2)

    assert args.num_workers == 0
    assert args.persistent_workers is False


def test_windows_workers_can_be_explicitly_enabled(monkeypatch):
    monkeypatch.setattr(acpr_bddx_data.os, "name", "nt", raising=False)

    args = acpr_bddx_data.build_bddx_acpr_args(_base_cfg(True), split="train", batch_size=2)

    assert args.num_workers == 4
    assert args.persistent_workers is True
