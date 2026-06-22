from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Literal
import os

import torch
from torch.utils.data import DataLoader, Dataset

from fate_x.acpr_dynflow.types import DynFlowBatch


class SyntheticDynFlowDataset(Dataset):
    def __init__(self, length: int = 8, frames: int = 32, vocab: int = 30522):
        self.length = length
        self.frames = frames
        self.vocab = vocab

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        g = torch.Generator().manual_seed(idx)
        control = torch.randn(self.frames, 2, generator=g)
        return {
            "frames": torch.randn(self.frames, 3, 224, 224, generator=g),
            "input_ids": torch.randint(0, self.vocab, (30,), generator=g),
            "attention_mask": torch.ones(30, dtype=torch.long),
            "token_type_ids": torch.zeros(30, dtype=torch.long),
            "masked_pos": torch.arange(0, 8, dtype=torch.long),
            "masked_ids": torch.randint(0, self.vocab, (8,), generator=g),
            "control_target": control,
            "sample_id": f"synthetic_dynflow_{idx}",
            "raw_action": "vehicle moves through traffic",
            "raw_justification": "traffic flow changes ahead",
        }


def collate_dynflow(rows: list[dict[str, Any]]) -> DynFlowBatch:
    return DynFlowBatch(
        frames=torch.stack([r["frames"] for r in rows]),
        input_ids=torch.stack([r["input_ids"] for r in rows]),
        attention_mask=torch.stack([r["attention_mask"] for r in rows]),
        token_type_ids=torch.stack([r["token_type_ids"] for r in rows]),
        masked_pos=torch.stack([r["masked_pos"] for r in rows]),
        masked_ids=torch.stack([r["masked_ids"] for r in rows]),
        control_target=torch.stack([r["control_target"] for r in rows]),
        sample_ids=[r["sample_id"] for r in rows],
        raw_actions=[r["raw_action"] for r in rows],
        raw_justifications=[r["raw_justification"] for r in rows],
    )


def _import_bert_tokenizer():
    try:
        from src.layers.bert.tokenization_bert import BertTokenizer
    except Exception:
        from transformers import BertTokenizer
    return BertTokenizer


def build_adapt_args(cfg: dict[str, Any], split: str, batch_size: int, max_samples: int = -1) -> tuple[Any, Any, str]:
    from src.layers.bert.tokenization_bert import BertTokenizer
    data = cfg.get("data", {})
    paths = cfg.get("paths", {})
    yaml_key = "train_yaml" if split == "train" else ("test_caption_yaml" if split == "test" else "test_caption_yaml")
    yaml_path = paths[yaml_key]
    norm = str(yaml_path).replace("\\", "/")
    data_dir = "datasets_part" if norm.startswith("datasets_part/") else "datasets"
    rel_yaml = norm[len(data_dir) + 1 :] if norm.startswith(data_dir + "/") else norm
    workers = int(data.get("num_workers", 0))
    if os.name == "nt":
        workers = 0
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
        signal_types=["course", "speed"],
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
        per_gpu_train_batch_size=batch_size,
        per_gpu_eval_batch_size=batch_size,
        num_train_epochs=1,
        num_workers=workers,
        seed=88,
        num_gpus=1,
        limited_samples=max_samples if split == "train" else -1,
        limited_eval_samples=max_samples if split != "train" else -1,
        debug_speed=False,
        acpr_flow_preserve_meta=True,
    )
    tokenizer = BertTokenizer.from_pretrained(paths.get("bert_dir", "models/captioning/bert-base-uncased"), do_lower_case=True)
    return args, tokenizer, rel_yaml


def adapt_batch_to_dynflow(batch: Any) -> DynFlowBatch:
    if isinstance(batch, DynFlowBatch):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        keys, examples, meta = batch
        input_ids = examples[0]
        attention_mask = examples[1]
        token_type_ids = examples[2]
        frames = examples[3]
        masked_pos = examples[4] if len(examples) > 4 and torch.is_tensor(examples[4]) else None
        if len(examples) >= 7:
            masked_ids = examples[5]
            control = examples[-1]
        else:
            masked_ids = None
            control = examples[-1]
        sample_ids = list(meta.get("sample_id", keys)) if isinstance(meta, dict) else list(keys)
        raw_actions = list(meta.get("raw_action", [""] * len(sample_ids))) if isinstance(meta, dict) else [""] * len(sample_ids)
        raw_justifications = list(meta.get("raw_justification", [""] * len(sample_ids))) if isinstance(meta, dict) else [""] * len(sample_ids)
        if control is not None and control.ndim == 3 and control.shape[1] == 2:
            control = control.transpose(1, 2)
        return DynFlowBatch(frames=frames, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, masked_pos=masked_pos, masked_ids=masked_ids, control_target=control, sample_ids=sample_ids, raw_actions=raw_actions, raw_justifications=raw_justifications)
    raise TypeError(f"unsupported batch type {type(batch)!r}")


class DynFlowLoaderAdapter:
    def __init__(self, loader: Iterable[Any]):
        self.loader = loader
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self):
        for batch in self.loader:
            yield adapt_batch_to_dynflow(batch)

    def __len__(self):
        return len(self.loader)  # type: ignore[arg-type]


def build_dynflow_dataloader(cfg: dict[str, Any], split: Literal["train", "test"], batch_size: int = 1, max_samples: int = -1, synthetic: bool = False):
    if synthetic:
        return DataLoader(SyntheticDynFlowDataset(length=max_samples if max_samples > 0 else 8), batch_size=batch_size, collate_fn=collate_dynflow)
    from src.datasets.vl_dataloader import make_data_loader
    args, tokenizer, yaml_path = build_adapt_args(cfg, split, batch_size, max_samples)
    loader = make_data_loader(args, yaml_path, tokenizer, is_distributed=False, is_train=(split == "train"), num_gpus=1)
    return DynFlowLoaderAdapter(loader)

