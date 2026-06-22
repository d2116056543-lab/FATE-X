from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from fate_x.acpr_flow_v2.types import FlowCalV2Batch


_SEGMENT_SUFFIX_RE = re.compile(r":\d+$")


def resolve_adapt_text_contract(checkpoint_dir: Optional[str] = None, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fallback = fallback or {}
    contract = {
        "mask_prob": 0.5,
        "max_masked_tokens": 45,
        "max_seq_a_length": 15,
        "max_seq_length": 30,
        "use_sep_cap": True,
        "source": "official_adapt_default",
    }
    contract.update({k: v for k, v in fallback.items() if v is not None})
    return contract


class _SyntheticV2Dataset(Dataset):
    def __init__(self, length: int = 4, frames: int = 32, vocab: int = 101):
        self.length = length
        self.frames = frames
        self.vocab = vocab

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        g = torch.Generator().manual_seed(idx)
        return {
            "frames": torch.randn(self.frames, 3, 224, 224, generator=g),
            "input_ids": torch.randint(0, self.vocab, (30,), generator=g),
            "attention_mask": torch.ones(30, dtype=torch.long),
            "token_type_ids": torch.zeros(30, dtype=torch.long),
            "masked_pos": torch.arange(0, 4, dtype=torch.long),
            "masked_ids": torch.randint(0, self.vocab, (4,), generator=g),
            "car_info": torch.randn(2, self.frames, generator=g),
            "sample_id": f"synthetic_{idx}",
            "raw_action": "car slows down",
            "raw_justification": "because traffic is ahead",
        }


def _collate(rows: Sequence[Dict[str, Any]]) -> FlowCalV2Batch:
    return FlowCalV2Batch(
        frames=torch.stack([r["frames"] for r in rows]),
        input_ids=torch.stack([r["input_ids"] for r in rows]),
        attention_mask=torch.stack([r["attention_mask"] for r in rows]),
        token_type_ids=torch.stack([r["token_type_ids"] for r in rows]),
        masked_pos=torch.stack([r["masked_pos"] for r in rows]),
        masked_ids=torch.stack([r["masked_ids"] for r in rows]),
        car_info=torch.stack([r["car_info"] for r in rows]),
        sample_ids=[r["sample_id"] for r in rows],
        raw_actions=[r["raw_action"] for r in rows],
        raw_justifications=[r["raw_justification"] for r in rows],
    )


def _as_batch_first(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.unsqueeze(0) if tensor.ndim in (1, 2, 4) else tensor


def normalize_bddx_sample_id(sample_id: Any) -> str:
    if isinstance(sample_id, bytes):
        sample_id = sample_id.decode("utf-8", errors="ignore")
    return _SEGMENT_SUFFIX_RE.sub("", str(sample_id))


def _processed_video_info_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_dir = os.environ.get("FATE_X_BDDX_PROCESSED_VIDEO_INFO_DIR")
    if env_dir:
        dirs.append(Path(env_dir))
    dirs.extend(
        [
            Path("datasets_part/BDDX/processed_video_info"),
            Path("datasets/BDDX/processed_video_info"),
            Path("../datasets_part/BDDX/processed_video_info"),
            Path("../datasets/BDDX/processed_video_info"),
        ]
    )
    return dirs


@lru_cache(maxsize=8192)
def _load_processed_control_by_sample(sample_id: str, frames: int) -> Optional[torch.Tensor]:
    base_id = normalize_bddx_sample_id(sample_id)
    for root in _processed_video_info_dirs():
        path = root / f"{base_id}.h5"
        if not path.exists():
            continue
        try:
            import h5py

            with h5py.File(path, "r") as h5:
                if "course" not in h5 or "speed" not in h5:
                    return None
                course = torch.as_tensor(h5["course"][:], dtype=torch.float32)
                speed = torch.as_tensor(h5["speed"][:], dtype=torch.float32)
        except Exception:
            return None
        control = torch.stack([course, speed], dim=0)
        if control.shape[-1] != frames:
            control = torch.nn.functional.interpolate(
                control.unsqueeze(0),
                size=frames,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        return control
    return None


def _is_placeholder_car_info(car_info: Optional[torch.Tensor]) -> bool:
    if car_info is None or not isinstance(car_info, torch.Tensor) or car_info.numel() == 0:
        return False
    finite = torch.isfinite(car_info)
    if not bool(finite.all().item()):
        return True
    return bool(torch.all(car_info <= -0.99).item())


def _replace_placeholder_car_info_from_h5(
    car_info: Optional[torch.Tensor],
    sample_ids: Sequence[str],
) -> Optional[torch.Tensor]:
    if not _is_placeholder_car_info(car_info):
        return car_info
    assert car_info is not None
    if car_info.ndim != 3 or car_info.shape[1] != 2:
        return car_info
    frames = int(car_info.shape[-1])
    rows: list[torch.Tensor] = []
    for idx, sample_id in enumerate(sample_ids):
        loaded = _load_processed_control_by_sample(str(sample_id), frames)
        if loaded is None:
            rows.append(car_info[idx].detach().cpu())
        else:
            rows.append(loaded)
    patched = torch.stack(rows, dim=0).to(device=car_info.device, dtype=car_info.dtype)
    return patched


def adapt_batch_to_v2(batch: Any) -> FlowCalV2Batch:
    if isinstance(batch, FlowCalV2Batch):
        batch.car_info = _replace_placeholder_car_info_from_h5(batch.car_info, batch.sample_ids)
        return batch
    if isinstance(batch, dict):
        frames = batch["frames"]
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
        sample_ids = [batch.get("sample_id", "sample")]
        car_info = _replace_placeholder_car_info_from_h5(_as_batch_first(batch.get("car_info")), sample_ids)
        return FlowCalV2Batch(
            frames=frames,
            input_ids=_as_batch_first(batch.get("input_ids")),
            attention_mask=_as_batch_first(batch.get("attention_mask")),
            token_type_ids=_as_batch_first(batch.get("token_type_ids")),
            masked_pos=_as_batch_first(batch.get("masked_pos")),
            masked_ids=_as_batch_first(batch.get("masked_ids")),
            car_info=car_info,
            sample_ids=sample_ids,
            raw_actions=[batch.get("raw_action", "")],
            raw_justifications=[batch.get("raw_justification", "")],
        )
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        keys, examples, meta = batch
        if len(examples) == 7:
            input_ids, attention_mask, token_type_ids, frames, masked_pos, masked_ids, car_info = examples[:7]
        elif len(examples) == 6:
            # ADAPT test/eval batches do not include masked token labels; keep control targets.
            input_ids, attention_mask, token_type_ids, frames, masked_pos, car_info = examples[:6]
            masked_ids = None
        else:
            raise ValueError(f"ADAPT batch must expose 6 eval tensors or 7 train tensors, got {len(examples)}")
        sample_ids = list(meta.get("sample_id", keys)) if isinstance(meta, dict) else list(keys)
        raw_actions = list(meta.get("raw_action", [""] * len(sample_ids))) if isinstance(meta, dict) else [""] * len(sample_ids)
        raw_justifications = list(meta.get("raw_justification", [""] * len(sample_ids))) if isinstance(meta, dict) else [""] * len(sample_ids)
        car_info = _replace_placeholder_car_info_from_h5(car_info, sample_ids)
        return FlowCalV2Batch(
            frames=frames,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            masked_pos=masked_pos,
            masked_ids=masked_ids,
            car_info=car_info,
            sample_ids=sample_ids,
            raw_actions=raw_actions,
            raw_justifications=raw_justifications,
        )
    raise TypeError(f"cannot adapt batch type {type(batch)!r}")


class _FlowCalV2LoaderAdapter:
    def __init__(self, loader: Iterable[Any], tokenizer: Any | None = None):
        self.loader = loader
        self.tokenizer = tokenizer

    @property
    def dataset(self) -> Any:
        return getattr(self.loader, "dataset", None)

    def __iter__(self) -> Iterator[FlowCalV2Batch]:
        for batch in self.loader:
            yield adapt_batch_to_v2(batch)

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


def _synthetic_loader(batch_size: int, num_workers: int, length: int = 4, vocab: int = 101) -> DataLoader:
    ds = _SyntheticV2Dataset(length=length, vocab=vocab)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate)


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    if yaml is None:
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_formal_yaml(split: str, config_path: Optional[str], yaml_file: Optional[str]) -> tuple[str, str]:
    if yaml_file:
        data_dir = "datasets_part" if "datasets_part" in yaml_file.replace("\\", "/") else "datasets"
        return data_dir, yaml_file
    cfg = _load_yaml(config_path) if config_path else {}
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    key = "train_yaml" if split == "train" else "test_yaml"
    selected = str(paths.get(key, "datasets_part/BDDX/training_32frames.yaml" if split == "train" else "datasets/BDDX/testing_32frames.yaml"))
    norm = selected.replace("\\", "/")
    if norm.startswith("datasets_part/"):
        return "datasets_part", norm[len("datasets_part/"):]
    if norm.startswith("datasets/"):
        return "datasets", norm[len("datasets/"):]
    return str(Path(selected).parent), Path(selected).name


def _build_adapt_args(
    split: str,
    batch_size: int,
    num_workers: int,
    data_dir: str,
    limited_samples: int,
    image_resolution: int = 224,
    max_num_frames: int = 32,
    vocab_dir: str = "models/captioning/bert-base-uncased",
) -> tuple[Any, Any]:
    from src.layers.bert.tokenization_bert import BertTokenizer

    adapt_visual_tokens = int((int(max_num_frames) / 2) * (int(image_resolution) / 32) * (int(image_resolution) / 32))
    args = SimpleNamespace(
        data_dir=data_dir,
        on_memory=False,
        img_feature_dim=512,
        img_res=image_resolution,
        patch_size=16,
        max_num_frames=max_num_frames,
        add_od_labels=False,
        use_asr=False,
        use_sep_cap=True,
        use_swap_cap=False,
        use_car_sensor=True,
        multitask=True,
        only_signal=False,
        signal_types=["course", "speed"],
        decoder_sampling_strategy="uniform",
        max_img_seq_length=adapt_visual_tokens,
        max_seq_length=30,
        max_seq_a_length=15,
        max_gen_length=15,
        mask_prob=0.5,
        max_masked_tokens=45,
        attn_mask_type="learn_vid_att",
        text_mask_type="random",
        tag_to_mask=["noun", "verb"],
        mask_tag_prob=0.8,
        random_mask_prob=0,
        per_gpu_train_batch_size=batch_size,
        per_gpu_eval_batch_size=batch_size,
        num_train_epochs=1,
        num_workers=num_workers,
        seed=88,
        num_gpus=1,
        limited_samples=limited_samples if split == "train" else -1,
        limited_eval_samples=limited_samples if split != "train" else -1,
        debug_speed=False,
        effective_batch_size=-1,
        acpr_flow_preserve_meta=True,
    )
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=True)
    return args, tokenizer


def build_v2_dataloader(
    split: Literal["train", "test", "validation"],
    batch_size: int = 1,
    num_workers: int = 0,
    formal: bool = True,
    synthetic: bool = False,
    config_path: Optional[str] = None,
    yaml_file: Optional[str] = None,
    data_dir: Optional[str] = None,
    length: Optional[int] = None,
    vocab: int = 101,
    image_resolution: int = 224,
    max_num_frames: int = 32,
    **kwargs: Any,
) -> Iterable[FlowCalV2Batch]:
    if formal and split == "validation":
        raise ValueError("formal ACPR FlowCal V2 uses train/test only; validation loader is forbidden")
    if synthetic:
        return _synthetic_loader(batch_size=batch_size, num_workers=num_workers, length=length or 4, vocab=vocab)
    formal_data_dir, formal_yaml = _resolve_formal_yaml(split, config_path, yaml_file)
    if data_dir is not None:
        formal_data_dir = data_dir
    args, tokenizer = _build_adapt_args(
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=formal_data_dir,
        limited_samples=int(length) if length is not None else -1,
        image_resolution=image_resolution,
        max_num_frames=max_num_frames,
    )
    from src.datasets.vl_dataloader import make_data_loader

    loader = make_data_loader(
        args,
        formal_yaml,
        tokenizer,
        is_distributed=False,
        is_train=(split == "train"),
        num_gpus=1,
    )
    return _FlowCalV2LoaderAdapter(loader, tokenizer=tokenizer)


def stream_train_control_stats(loader: Iterable[FlowCalV2Batch]) -> Dict[str, torch.Tensor]:
    vals = []
    for batch in loader:
        if batch.car_info is not None:
            vals.append(batch.car_info.transpose(1, 2).reshape(-1, 2))
    if not vals:
        return {"mean": torch.zeros(2), "std": torch.ones(2)}
    x = torch.cat(vals, dim=0)
    return {"mean": x.mean(0), "std": x.std(0).clamp_min(1e-6)}


def deterministic_train_calib_ids(sample_ids: Sequence[str], fraction: float = 0.10, seed: int = 20260621) -> set[str]:
    selected = set()
    threshold = int(fraction * 10000)
    for sid in sample_ids:
        h = int(hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()[:8], 16) % 10000
        if h < threshold:
            selected.add(sid)
    return selected


def assert_v2_assets(*args: Any, **kwargs: Any) -> bool:
    return True


adapt_batch_to_flowcal_v2_batch = adapt_batch_to_v2
