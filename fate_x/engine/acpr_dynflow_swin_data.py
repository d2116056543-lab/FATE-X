from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Literal
import os

import torch
from torch.utils.data import DataLoader, Dataset

from fate_x.acpr_dynflow_swin.types import DynFlowSwinBatch


class SyntheticDynFlowSwinDataset(Dataset):
    def __init__(self, length: int = 8, frames: int = 32, image_size: int = 32, vocab: int = 30522):
        self.length = length
        self.frames = frames
        self.image_size = image_size
        self.vocab = vocab

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(idx)
        return {
            "frames": torch.randn(self.frames, 3, self.image_size, self.image_size, generator=generator),
            "input_ids": torch.randint(0, self.vocab, (30,), generator=generator),
            "attention_mask": torch.ones(30, dtype=torch.long),
            "token_type_ids": torch.zeros(30, dtype=torch.long),
            "masked_pos": torch.ones(30, dtype=torch.long),
            "masked_ids": torch.randint(0, self.vocab, (30,), generator=generator),
            "control_target": torch.randn(self.frames, 2, generator=generator),
            "sample_id": f"synthetic_dynflow_swin_{idx}",
            "raw_action": "vehicle moves through traffic",
            "raw_justification": "traffic flow changes ahead",
        }


def collate_dynflow_swin(rows: list[dict[str, Any]]) -> DynFlowSwinBatch:
    return DynFlowSwinBatch(
        frames=torch.stack([row["frames"] for row in rows]),
        input_ids=torch.stack([row["input_ids"] for row in rows]),
        attention_mask=torch.stack([row["attention_mask"] for row in rows]),
        token_type_ids=torch.stack([row["token_type_ids"] for row in rows]),
        masked_pos=torch.stack([row["masked_pos"] for row in rows]),
        masked_ids=torch.stack([row["masked_ids"] for row in rows]),
        control_target=torch.stack([row["control_target"] for row in rows]),
        sample_ids=[row["sample_id"] for row in rows],
        raw_actions=[row["raw_action"] for row in rows],
        raw_justifications=[row["raw_justification"] for row in rows],
    )


def _import_bert_tokenizer():
    try:
        from src.layers.bert.tokenization_bert import BertTokenizer
    except Exception:
        from transformers import BertTokenizer
    return BertTokenizer


def _yaml_for_split(cfg: dict[str, Any], split: str) -> str:
    paths = cfg.get("paths", {})
    if split == "train":
        return str(paths.get("train_yaml", "datasets_part/BDDX/training_32frames.yaml"))
    if split == "test_signal":
        return str(paths.get("test_signal_yaml", "datasets_part/BDDX/testing_32frames.yaml"))
    return str(paths.get("test_caption_yaml", "datasets/BDDX/testing_32frames.yaml"))


def build_adapt_compatible_args(
    cfg: dict[str, Any],
    split: str,
    batch_size: int,
    max_samples: int = -1,
) -> tuple[SimpleNamespace, Any, str]:
    data = cfg.get("data", {})
    paths = cfg.get("paths", {})
    yaml_path = _yaml_for_split(cfg, split).replace("\\", "/")
    data_dir = "datasets_part" if yaml_path.startswith("datasets_part/") else "datasets"
    rel_yaml = yaml_path[len(data_dir) + 1 :] if yaml_path.startswith(data_dir + "/") else yaml_path
    workers = int(data.get("num_workers", 0))
    persistent_workers = bool(data.get("persistent_workers", True))
    if os.name == "nt":
        workers = 0
        persistent_workers = False
    if workers <= 0:
        persistent_workers = False

    args = SimpleNamespace(
        data_dir=data_dir,
        on_memory=False,
        img_feature_dim=512,
        img_res=int(data.get("image_resolution", 224)),
        patch_size=16,
        max_num_frames=int(data.get("frames", 32)),
        add_od_labels=False,
        use_asr=False,
        use_sep_cap=bool(data.get("use_sep_cap", True)),
        use_swap_cap=False,
        use_car_sensor=True,
        multitask=True,
        only_signal=False,
        signal_types=list(data.get("signal_names", ["course", "speed"])),
        max_img_seq_length=784,
        max_seq_length=int(data.get("max_seq_length", 30)),
        max_seq_a_length=int(data.get("action_max_tokens", 15)),
        max_gen_length=int(data.get("explanation_max_tokens", 15)),
        mask_prob=float(data.get("mask_prob", 0.5)),
        max_masked_tokens=int(data.get("max_masked_tokens", 45)),
        attn_mask_type="learn_vid_att",
        text_mask_type="random",
        tag_to_mask=["noun", "verb"],
        mask_tag_prob=0.8,
        random_mask_prob=0,
        per_gpu_train_batch_size=int(batch_size),
        per_gpu_eval_batch_size=int(batch_size),
        num_train_epochs=1,
        num_workers=workers,
        pin_memory=bool(data.get("pin_memory", True)),
        persistent_workers=persistent_workers,
        prefetch_factor=int(data.get("prefetch_factor", 2)),
        seed=88,
        num_gpus=1,
        limited_samples=max_samples if split == "train" and max_samples > 0 else -1,
        limited_eval_samples=max_samples if split != "train" and max_samples > 0 else -1,
        debug_speed=False,
        acpr_flow_preserve_meta=True,
    )
    tokenizer_cls = _import_bert_tokenizer()
    tokenizer = tokenizer_cls.from_pretrained(
        paths.get("bert_dir", "models/captioning/bert-base-uncased"),
        do_lower_case=True,
    )
    return args, tokenizer, rel_yaml


def adapt_batch_to_dynflow_swin(batch: Any) -> DynFlowSwinBatch:
    if isinstance(batch, DynFlowSwinBatch):
        return batch
    if not (isinstance(batch, (list, tuple)) and len(batch) == 3):
        raise TypeError(f"unsupported BDD-X batch type {type(batch)!r}")
    keys, examples, meta = batch
    input_ids = examples[0]
    attention_mask = examples[1]
    token_type_ids = examples[2]
    frames = examples[3]
    masked_pos = examples[4] if len(examples) > 4 and torch.is_tensor(examples[4]) else torch.ones_like(input_ids)
    masked_ids = examples[5] if len(examples) > 5 and torch.is_tensor(examples[5]) else input_ids
    control = examples[-1] if torch.is_tensor(examples[-1]) else torch.zeros(frames.shape[0], frames.shape[1], 2)
    if control.ndim == 3 and control.shape[1] == 2:
        control = control.transpose(1, 2)
    if frames.ndim != 5 or frames.shape[1] != 32 or frames.shape[2] != 3:
        raise ValueError(f"BDD-X loader must emit frames [B,32,3,H,W], got {tuple(frames.shape)}")
    if control.ndim != 3 or control.shape[-1] != 2:
        raise ValueError(f"BDD-X loader must emit control target [B,T,2] or [B,2,T], got {tuple(control.shape)}")
    sample_ids = list(meta.get("sample_id", keys)) if isinstance(meta, dict) else list(keys)
    raw_actions = list(meta.get("raw_action", [""] * len(sample_ids))) if isinstance(meta, dict) else [""] * len(sample_ids)
    raw_justifications = (
        list(meta.get("raw_justification", [""] * len(sample_ids))) if isinstance(meta, dict) else [""] * len(sample_ids)
    )
    return DynFlowSwinBatch(
        frames=frames,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        masked_pos=masked_pos,
        masked_ids=masked_ids,
        control_target=control,
        sample_ids=sample_ids,
        raw_actions=raw_actions,
        raw_justifications=raw_justifications,
    )


class DynFlowSwinLoaderAdapter:
    def __init__(self, loader: Iterable[Any]):
        self.loader = loader
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self):
        for batch in self.loader:
            yield adapt_batch_to_dynflow_swin(batch)

    def __len__(self):
        return len(self.loader)  # type: ignore[arg-type]


def build_dynflow_swin_dataloader(
    cfg: dict[str, Any],
    split: Literal["train", "test", "test_signal"] = "train",
    batch_size: int = 1,
    max_samples: int = -1,
    synthetic: bool = False,
):
    if synthetic:
        length = max_samples if max_samples > 0 else 8
        dataset = SyntheticDynFlowSwinDataset(length=length)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_dynflow_swin)
    from src.datasets.vl_dataloader import make_data_loader

    args, tokenizer, yaml_path = build_adapt_compatible_args(cfg, split, batch_size, max_samples)
    loader = make_data_loader(
        args,
        yaml_path,
        tokenizer,
        is_distributed=False,
        is_train=(split == "train"),
        num_gpus=1,
    )
    return DynFlowSwinLoaderAdapter(loader)


def resolve_data_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = cfg.get("paths", {})
    return {
        "train": Path(paths.get("train_yaml", "datasets_part/BDDX/training_32frames.yaml")),
        "test_caption": Path(paths.get("test_caption_yaml", "datasets/BDDX/testing_32frames.yaml")),
        "test_signal": Path(paths.get("test_signal_yaml", "datasets_part/BDDX/testing_32frames.yaml")),
    }
