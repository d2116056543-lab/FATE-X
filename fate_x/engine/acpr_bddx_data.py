from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import os

import torch

from fate_x.acpr_flow.types import ACPRFlowBatch


def _import_bert_tokenizer():
    try:
        from src.layers.bert.tokenization_bert import BertTokenizer
    except Exception:
        try:
            from src.pytorch_transformers import BertTokenizer
        except Exception:
            from transformers import BertTokenizer
    return BertTokenizer


def build_bddx_acpr_args(cfg: dict[str, Any], split: str, batch_size: int, max_samples: int = -1) -> SimpleNamespace:
    data = cfg.get("data", {})
    paths = cfg.get("paths", {})
    image_resolution = int(data.get("image_resolution", 224))
    frames = int(data.get("max_num_frames", 32))
    limited_train = int(max_samples) if split == "train" and int(max_samples) > 0 else -1
    limited_eval = int(max_samples) if split != "train" and int(max_samples) > 0 else -1
    num_workers = int(data.get("num_workers", 4))
    persistent_workers = bool(data.get("persistent_workers", True))
    allow_windows_workers = bool(data.get("allow_windows_workers", False))
    if os.name == "nt" and not allow_windows_workers:
        num_workers = 0
        persistent_workers = False
    if num_workers <= 0:
        persistent_workers = False
    return SimpleNamespace(
        data_dir=".",
        max_num_frames=frames,
        img_res=image_resolution,
        vidswin_size="base",
        kinetics=600,
        pretrained_2d=False,
        pretrained_checkpoint="",
        grid_feat=True,
        patch_size=16,
        img_feature_dim=512,
        max_img_seq_length=max(1, (frames // 2) * (image_resolution // 32) * (image_resolution // 32)),
        max_seq_length=int(data.get("action_max_tokens", 15)) + int(data.get("explanation_max_tokens", 15)),
        max_seq_a_length=int(data.get("action_max_tokens", 15)),
        max_gen_length=int(data.get("explanation_max_tokens", 15)),
        decoder_num_frames=frames,
        use_car_sensor=True,
        multitask=True,
        only_signal=False,
        signal_types=list(data.get("signals", ["course", "speed"])),
        add_od_labels=False,
        use_sep_cap=bool(data.get("use_sep_cap", True)),
        use_swap_cap=False,
        use_asr=False,
        on_memory=False,
        debug_speed=False,
        do_lower_case=True,
        mask_prob=0.15,
        max_masked_tokens=20,
        random_mask_prob=0.8,
        mask_tag_prob=0.0,
        tag_to_mask=None,
        attn_mask_type="learn_vid_att",
        text_mask_type="random",
        num_workers=num_workers,
        pin_memory=bool(data.get("pin_memory", True)),
        persistent_workers=persistent_workers,
        per_gpu_train_batch_size=int(batch_size),
        per_gpu_eval_batch_size=int(batch_size),
        num_train_epochs=1,
        limited_samples=limited_train,
        limited_eval_samples=limited_eval,
        seed=88,
        num_gpus=1,
        acpr_flow_preserve_meta=True,
        bert_dir=paths.get("bert_dir", "models/captioning/bert-base-uncased"),
    )


def build_bddx_acpr_dataloader(
    cfg: dict[str, Any],
    split: str,
    batch_size: int,
    max_samples: int = -1,
    tokenizer=None,
    distributed: bool = False,
):
    from src.datasets.vl_dataloader import make_data_loader

    paths = cfg.get("paths", {})
    yaml_key = "train_yaml" if split == "train" else "test_yaml"
    yaml_path = paths[yaml_key]
    args = build_bddx_acpr_args(cfg, split=split, batch_size=batch_size, max_samples=max_samples)
    if tokenizer is None:
        tokenizer_cls = _import_bert_tokenizer()
        tokenizer = tokenizer_cls.from_pretrained(args.bert_dir, do_lower_case=True)
    return make_data_loader(
        args,
        yaml_path,
        tokenizer,
        is_distributed=distributed,
        is_train=(split == "train"),
        num_gpus=1,
    )


def adapt_batch_to_acpr_flow_batch(keys, examples, meta: dict[str, list], device: str | torch.device) -> ACPRFlowBatch:
    input_ids = examples[0].to(device) if len(examples) > 0 and torch.is_tensor(examples[0]) else None
    attention_mask = examples[1].to(device) if len(examples) > 1 and torch.is_tensor(examples[1]) else None
    token_type_ids = examples[2].to(device) if len(examples) > 2 and torch.is_tensor(examples[2]) else None
    frames = examples[3].to(device, non_blocking=True)
    masked_pos = examples[4].to(device) if len(examples) > 4 and torch.is_tensor(examples[4]) else None
    masked_ids = examples[5].to(device) if len(examples) > 5 and torch.is_tensor(examples[5]) else None
    car_info = examples[-1].to(device) if len(examples) > 0 and torch.is_tensor(examples[-1]) else None
    raw_actions = list(meta.get("raw_action", [""] * len(keys)))
    raw_justifications = list(meta.get("raw_justification", [""] * len(keys)))
    sample_ids = list(meta.get("sample_id", keys))
    if frames.ndim != 5 or frames.shape[1] != 32 or frames.shape[2] != 3:
        raise ValueError(f"BDD-X ACPR dataloader must emit frames [B,32,3,H,W], got {tuple(frames.shape)}")
    return ACPRFlowBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        frames=frames,
        masked_pos=masked_pos,
        masked_ids=masked_ids,
        car_info=car_info,
        sample_ids=sample_ids,
        raw_actions=raw_actions,
        raw_justifications=raw_justifications,
    )


def assert_required_assets(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for key in ("train_yaml", "test_yaml", "adapt_checkpoint", "bert_dir", "video_swin_checkpoint"):
        path = Path(cfg["paths"][key])
        report[key] = {"path": str(path), "exists": path.exists()}
    missing = [key for key, item in report.items() if not item["exists"]]
    if missing:
        raise FileNotFoundError(f"Missing ACPR formal assets: {missing}; report={report}")
    return report
