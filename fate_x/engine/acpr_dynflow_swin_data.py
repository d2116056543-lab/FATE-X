from __future__ import annotations

from pathlib import Path


def resolve_data_paths(cfg: dict) -> dict[str, Path]:
    paths = cfg.get("paths", {})
    return {
        "train": Path(paths.get("train_yaml", "datasets_part/BDDX/training_32frames.yaml")),
        "test_caption": Path(paths.get("test_caption_yaml", "datasets/BDDX/testing_32frames.yaml")),
        "test_signal": Path(paths.get("test_signal_yaml", "datasets_part/BDDX/testing_32frames.yaml")),
    }
