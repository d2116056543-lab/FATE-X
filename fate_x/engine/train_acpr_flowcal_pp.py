from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fate_x.acpr_flow.model import ACPRFlowModel, ACPRFlowModelConfig
from fate_x.acpr_flow.online_reason_target import build_action_residual_reason_target
from fate_x.acpr_flow.region_priors import ACPR_PREDICATE_NAMES, FLOW_FACTOR_NAMES
from fate_x.acpr_flow.sequence_calalign import SequenceCalAlign
from fate_x.engine.acpr_action_text_eval import evaluate_action_text_decisions
from fate_x.engine.acpr_control_eval import compute_control_metrics
from fate_x.losses.acpr_flowcal_losses import is_circular_control_signal, signed_circular_delta_deg
from fate_x.engine.acpr_bddx_data import (
    _import_bert_tokenizer,
    adapt_batch_to_acpr_flow_batch,
    assert_required_assets,
    build_bddx_acpr_args,
    build_bddx_acpr_dataloader,
)
from fate_x.utils.acpr_flow_config import load_acpr_flow_config
from fate_x.utils.acpr_flow_artifacts import write_json
from src.layers.bert import BertForImageCaptioning
from src.modeling.load_bert import get_bert_model


def build_acpr_optimizer_groups(model: ACPRFlowModel) -> tuple[list[dict], dict[str, str]]:
    lr_by_prefix = {
        "predicate_field": 1e-4,
        "flow_composer": 1e-4,
        "reason_memory": 1e-4,
        "temporal_seca": 5e-5,
        "reason_control_adapter": 2e-5,
        "prefix_future_head": 5e-5,
        "hardpair": 5e-5,
        "backbone": 5e-6,
        "control_base": 1e-5,
        "control_hidden": 1e-5,
    }
    groups: dict[float, dict] = {}
    manifest = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("captioning_model.bert.encoder.layer.10") or name.startswith("captioning_model.bert.encoder.layer.11"):
            matched = "bert_last2"
            lr = 1e-5
        elif name.startswith("captioning_model"):
            matched = "bert_last2"
            lr = 1e-5
        elif name.startswith("hardpair"):
            matched = "hardpair_projection"
            lr = lr_by_prefix["hardpair"]
        else:
            matched = next((p for p in lr_by_prefix if name.startswith(p)), "new_modules")
            lr = lr_by_prefix.get(matched, 1e-4)
        groups.setdefault(lr, {"params": [], "lr": lr, "names": []})
        groups[lr]["params"].append(param)
        groups[lr]["names"].append(name)
        manifest[name] = matched
    seen = sum(len(group["names"]) for group in groups.values())
    if seen != len(manifest):
        raise RuntimeError("Optimizer group manifest contains duplicate parameters")
    return list(groups.values()), manifest


def build_formal_experiment_suite(config: str | dict[str, Any]) -> list[dict[str, Any]]:
    cfg = load_acpr_flow_config(config) if isinstance(config, str) else config
    suite = cfg.get("experiment_suite", {})
    return [
        {"name": "run0_adapt_baseline_eval", "epochs": 0, "kind": "baseline_eval", "metric_based_early_stop": False},
        {
            "name": "common_stage_a_acpr_x",
            "epochs": int(suite.get("common_stage_a", {}).get("epochs", 6)),
            "kind": "train",
            "flow_enabled": False,
            "transport_enabled": False,
            "prefix_future_enabled": False,
            "metric_based_early_stop": False,
        },
        {
            "name": "fork_b1_acpr_x_equal_budget",
            "epochs": int(suite.get("fork_b1_acpr_x", {}).get("train_epochs", 12)),
            "kind": "train",
            "flow_enabled": False,
            "transport_enabled": False,
            "prefix_future_enabled": False,
            "metric_based_early_stop": False,
        },
        {
            "name": "fork_b2_acpr_flowcal_pp",
            "epochs": int(suite.get("fork_b2_acpr_flowcal_pp", {}).get("train_epochs", 12)),
            "kind": "train",
            "flow_enabled": True,
            "transport_enabled": True,
            "prefix_future_enabled": True,
            "metric_based_early_stop": False,
        },
        {"name": "sequence_calalign_b1", "epochs": int(suite.get("fork_b1_acpr_x", {}).get("calibration_epochs", 3)), "kind": "calibration", "metric_based_early_stop": False},
        {"name": "sequence_calalign_b2", "epochs": int(suite.get("fork_b2_acpr_flowcal_pp", {}).get("calibration_epochs", 3)), "kind": "calibration", "metric_based_early_stop": False},
        {"name": "no_retrain_interventions", "epochs": 0, "kind": "interventions", "metric_based_early_stop": False},
        {"name": "canvas_generation", "epochs": 0, "kind": "canvas", "metric_based_early_stop": False},
        {"name": "dataset_atlas", "epochs": 0, "kind": "atlas", "metric_based_early_stop": False},
    ]


def build_model_config(cfg: dict[str, Any], load_pretrained_backbone: bool = True) -> ACPRFlowModelConfig:
    model_cfg = cfg.get("model", {})
    paths = cfg.get("paths", {})
    multiscale = model_cfg.get("multiscale", {})
    loss_cfg = cfg.get("loss", {})
    loss_weights = {
        "action_text": float(loss_cfg.get("action_text", 1.0)),
        "explanation_text": float(loss_cfg.get("explanation_text", 1.0)),
        "control": float(loss_cfg.get("control", 0.05)),
        "predicate_pu": float(loss_cfg.get("predicate_pu", 0.05)),
        "flow_pu": float(loss_cfg.get("flow_pu", 0.03)),
        "reason_semantic": float(loss_cfg.get("reason_semantic", 0.05)),
        "future_control": float(loss_cfg.get("future_control", 0.02)),
        "memory_diversity": float(loss_cfg.get("memory_diversity", 0.001)),
    }
    return ACPRFlowModelConfig(
        state_dim=int(model_cfg.get("state_dim", 256)),
        text_hidden_dim=int(model_cfg.get("text_hidden_dim", 768)),
        num_frames=int(cfg.get("data", {}).get("max_num_frames", 32)),
        image_resolution=int(cfg.get("data", {}).get("image_resolution", 224)),
        formal_backbone=True,
        load_pretrained_backbone=load_pretrained_backbone,
        video_swin_checkpoint=paths.get("video_swin_checkpoint"),
        bert_img_feature_dim=int(cfg.get("data", {}).get("img_feature_dim", 512)),
        fine_stage=int(multiscale.get("fine_stage", 2)),
        coarse_stage=int(multiscale.get("coarse_stage", 3)),
        use_transport=True,
        use_flow=True,
        use_prefix_future=True,
        invalid_control_value=float(cfg.get("data", {}).get("invalid_control_value", -1.0)),
        control_signal_names=tuple(str(s) for s in cfg.get("data", {}).get("signals", ["course", "speed"])),
        hardpair_queue_size=int(model_cfg.get("hardpair", {}).get("queue_size", 4096)),
        hardpair_margin=float(model_cfg.get("hardpair", {}).get("margin", 0.20)),
        hardpair_max_pairs_per_batch=int(model_cfg.get("hardpair", {}).get("max_pairs_per_batch", 64)),
        hardpair_pair_weight=float(model_cfg.get("hardpair", {}).get("pair_weight", 0.03)),
        hardpair_pair_budget_ratio=float(model_cfg.get("hardpair", {}).get("pair_budget_ratio", 0.08)),
        loss_weights=loss_weights,
    )


def load_adapt_trans_encoder_weights(captioning_model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load ADAPT's fine-tuned text generation head into BertForImageCaptioning."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported ADAPT checkpoint state type: {type(state)!r}")
    prefix = "trans_encoder."
    trans_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not trans_state:
        raise KeyError(f"No {prefix!r} weights found in ADAPT checkpoint: {checkpoint_path}")
    incompatible = captioning_model.load_state_dict(trans_state, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_key_count": len(trans_state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _adapt_checkpoint_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported ADAPT checkpoint state type: {type(state)!r}")
    return state


def load_adapt_visual_path_weights(model: ACPRFlowModel, checkpoint_path: str | Path) -> dict[str, Any]:
    """Initialize ACPR's formal visual-to-text path from the same ADAPT checkpoint used for eval.

    ADAPT's caption decoder was trained on `swin(images) -> fc -> BertForImageCaptioning`.
    Keeping the captioner while leaving Swin/fc random changes that input distribution and
    can collapse the second sep-cap explanation stream to empty strings.
    """
    state = _adapt_checkpoint_state(checkpoint_path)
    report: dict[str, Any] = {"checkpoint": str(checkpoint_path)}

    video_swin = getattr(getattr(model, "backbone", None), "video_swin", None)
    swin_state = {key[len("swin."):]: value for key, value in state.items() if key.startswith("swin.")}
    if video_swin is not None and swin_state:
        incompatible = video_swin.load_state_dict(swin_state, strict=False)
        report["swin_loaded_key_count"] = len(swin_state)
        report["swin_missing_keys"] = list(incompatible.missing_keys)
        report["swin_unexpected_keys"] = list(incompatible.unexpected_keys)
    else:
        report["swin_loaded_key_count"] = 0
        report["swin_missing_keys"] = []
        report["swin_unexpected_keys"] = []
        report["swin_reason"] = "missing model.backbone.video_swin or swin.* checkpoint keys"

    fc_weight = state.get("fc.weight")
    fc_bias = state.get("fc.bias")
    if fc_weight is not None and fc_bias is not None:
        device = next(model.parameters()).device
        dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
        in_features = int(fc_weight.shape[1])
        # Materialize LazyLinear before loading ADAPT's trained projection.
        model.bert_img_proj(torch.zeros(1, 1, in_features, device=device, dtype=dtype))
        model.bert_img_proj.load_state_dict({"weight": fc_weight, "bias": fc_bias}, strict=True)
        report["fc_loaded"] = True
        report["fc_weight_shape"] = list(fc_weight.shape)
    else:
        report["fc_loaded"] = False
        report["fc_weight_shape"] = None
    return report


def build_formal_captioning_model(cfg: dict[str, Any], device: str | torch.device) -> BertForImageCaptioning:
    """Load the original ADAPT BertForImageCaptioning used by the formal text path."""
    data = cfg.get("data", {})
    args = SimpleNamespace(
        config_name="",
        model_name_or_path=cfg["paths"]["bert_dir"],
        tokenizer_name="",
        do_lower_case=True,
        drop_out=0.1,
        tie_weights=True,
        freeze_embedding=False,
        label_smoothing=0.0,
        drop_worst_ratio=0.0,
        drop_worst_after=0,
        img_feature_dim=int(data.get("img_feature_dim", 512)),
        num_hidden_layers=-1,
        hidden_size=-1,
        num_attention_heads=-1,
        intermediate_size=-1,
        load_partial_weights=True,
    )
    model, _, _ = get_bert_model(args)
    load_report = load_adapt_trans_encoder_weights(model, cfg["paths"]["adapt_checkpoint"])
    model.acpr_adapt_trans_encoder_load_report = load_report
    return model.to(device)


def load_formal_checkpoints(cfg: dict[str, Any]) -> dict[str, Any]:
    assets = assert_required_assets(cfg)
    adapt_checkpoint = cfg["paths"]["adapt_checkpoint"]
    video_swin_checkpoint = cfg["paths"]["video_swin_checkpoint"]
    report = {
        "assets": assets,
        "adapt_checkpoint": adapt_checkpoint,
        "video_swin_checkpoint": video_swin_checkpoint,
        "adapt_checkpoint_loadable": False,
        "video_swin_checkpoint_loadable": False,
    }
    report["adapt_checkpoint_loadable"] = isinstance(torch.load(adapt_checkpoint, map_location="cpu"), (dict, torch.Tensor))
    report["video_swin_checkpoint_loadable"] = isinstance(torch.load(video_swin_checkpoint, map_location="cpu"), (dict, torch.Tensor))
    return report


def load_bert_word_embedding_weight(bert_dir: str, device: str | torch.device = "cpu") -> torch.Tensor:
    """Load the local BERT word embedding table used for online reason targets."""
    model_path = Path(bert_dir) / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing BERT checkpoint for online reason target: {model_path}")
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    candidates = [
        "bert.embeddings.word_embeddings.weight",
        "module.bert.embeddings.word_embeddings.weight",
        "embeddings.word_embeddings.weight",
    ]
    weight = next((state[key] for key in candidates if isinstance(state, dict) and key in state), None)
    if weight is None and isinstance(state, dict):
        for key, value in state.items():
            if key.endswith("embeddings.word_embeddings.weight") and torch.is_tensor(value):
                weight = value
                break
    if weight is None or not torch.is_tensor(weight):
        raise KeyError(f"Could not find BERT word embedding weight in {model_path}")
    return weight.detach().to(device=device, dtype=torch.float32)


def _texts_to_padded_embeddings(
    texts: list[str],
    tokenizer,
    word_embedding_weight: torch.Tensor,
    max_tokens: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded: list[list[int]] = []
    for text in texts:
        tokens = tokenizer.tokenize(str(text or ""))[:max_tokens]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(ids, int):
            ids = [ids]
        encoded.append([int(x) for x in ids])
    length = max(1, max((len(ids) for ids in encoded), default=0))
    device = word_embedding_weight.device
    ids_tensor = torch.zeros(len(encoded), length, dtype=torch.long, device=device)
    mask = torch.zeros(len(encoded), length, dtype=torch.bool, device=device)
    vocab_size = int(word_embedding_weight.shape[0])
    for row, ids in enumerate(encoded):
        clipped = [min(max(token_id, 0), vocab_size - 1) for token_id in ids[:length]]
        if not clipped:
            continue
        ids_tensor[row, : len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=device)
        mask[row, : len(clipped)] = True
    return F.embedding(ids_tensor, word_embedding_weight), mask


def build_reason_semantic_target_for_batch(
    raw_actions: list[str],
    raw_justifications: list[str],
    tokenizer,
    word_embedding_weight: torch.Tensor,
    device: str | torch.device,
    residual_strength: float = 1.0,
) -> torch.Tensor:
    """Build the formal online Gram-Schmidt reason target from current batch text.

    The target is detached and used only as training supervision; it is not an
    inference input and no text embedding cache is saved.
    """
    word_embedding_weight = word_embedding_weight.detach().to(device=device, dtype=torch.float32)
    action_embeddings, action_mask = _texts_to_padded_embeddings(raw_actions, tokenizer, word_embedding_weight)
    reason_embeddings, reason_mask = _texts_to_padded_embeddings(raw_justifications, tokenizer, word_embedding_weight)
    combined_embeddings = torch.cat([action_embeddings, reason_embeddings], dim=1)
    action_token_mask = torch.cat([action_mask, torch.zeros_like(reason_mask)], dim=1)
    reason_token_mask = torch.cat([torch.zeros_like(action_mask), reason_mask], dim=1)
    return build_action_residual_reason_target(
        combined_embeddings,
        action_token_mask,
        reason_token_mask,
        residual_strength=residual_strength,
    ).detach()




def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write torch checkpoints through a same-directory temp file, then replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    tmp.replace(target)


def save_training_checkpoint(
    output_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch_idx: int,
    global_step: int,
    optimizer_step: int,
    gradient_accumulation_steps: int,
    update_best: bool = False,
    eval_report: dict[str, Any] | None = None,
    epoch_complete: bool = False,
    best_checkpoint_names: list[str] | None = None,
) -> None:
    """Save latest every time and selected best checkpoints for distinct metrics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch_idx),
        "epoch_complete": bool(epoch_complete),
        "global_step": int(global_step),
        "optimizer_step": int(optimizer_step),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "eval_report": eval_report,
    }
    atomic_torch_save(payload, out / "checkpoint_latest.pth")
    if update_best:
        names = list(best_checkpoint_names or ["checkpoint_best_test.pth"])
        kind_by_name = {
            "checkpoint_best_test.pth": "best_test_model_only",
            "checkpoint_best_text.pth": "best_text_model_only",
            "checkpoint_best_control.pth": "best_control_model_only",
            "checkpoint_best_adapt_joint.pth": "best_adapt_joint_model_only",
        }
        for checkpoint_name in dict.fromkeys(names):
            best_payload = dict(payload)
            # Keep latest as the exact resume artifact. Best checkpoints are
            # metric-selection snapshots for inference/audit, so they stay model-only.
            best_payload["optimizer"] = None
            best_payload["checkpoint_kind"] = kind_by_name.get(str(checkpoint_name), "best_model_only")
            best_payload["best_checkpoint_model_only"] = True
            atomic_torch_save(best_payload, out / str(checkpoint_name))


def load_resume_state(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: str | torch.device,
) -> dict[str, int | bool | str]:
    """Load model/optimizer and return counters needed to continue training."""
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    optimizer_payload = checkpoint.get("optimizer")
    optimizer_loaded = False
    if optimizer is not None and optimizer_payload is not None:
        optimizer.load_state_dict(optimizer_payload)
        optimizer_loaded = True
    has_epoch = "epoch" in checkpoint
    epoch = int(checkpoint.get("epoch", -1))
    epoch_complete = bool(checkpoint.get("epoch_complete", True))
    return {
        "checkpoint": str(path),
        "start_epoch": epoch + 1 if epoch_complete else max(epoch, 0),
        "global_step": int(checkpoint.get("global_step", 0)),
        "optimizer_step": int(checkpoint.get("optimizer_step", 0)),
        "epoch_complete": epoch_complete,
        "has_epoch": has_epoch,
        "optimizer_loaded": optimizer_loaded,
    }


def resolve_resume_start_epoch(resume_state: dict[str, Any] | None, train_loader_len: int) -> int:
    """Resolve start epoch without moving backwards from completed global steps."""
    if not resume_state:
        return 0
    global_step = int(resume_state.get("global_step", 0))
    train_loader_len = max(1, int(train_loader_len))
    inferred_from_steps = global_step // train_loader_len
    if bool(resume_state.get("has_epoch", False)):
        return max(int(resume_state["start_epoch"]), inferred_from_steps)
    return inferred_from_steps


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean_finite(values: list[float | None]) -> float | None:
    finite_values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


def compute_adapt_selection_scores(eval_report: dict[str, Any]) -> dict[str, float]:
    """Return checkpoint-selection scores using only ADAPT-legal metrics."""
    text_cider = _finite_float_or_none(eval_report.get("metric_value"))
    control = eval_report.get("control_metrics") if isinstance(eval_report.get("control_metrics"), dict) else {}
    signals = control.get("signals", {}) if isinstance(control, dict) else {}
    speed = signals.get("speed", {}) if isinstance(signals.get("speed", {}), dict) else {}
    course = signals.get("course", {}) if isinstance(signals.get("course", {}), dict) else {}
    speed_rmse = _finite_float_or_none(speed.get("rmse"))
    course_rmse = _finite_float_or_none(course.get("rmse"))
    rmse_sum = None
    if speed_rmse is not None and course_rmse is not None:
        rmse_sum = speed_rmse + course_rmse
    threshold_mean = _mean_finite([
        _finite_float_or_none(speed.get("acc_at_0.5")),
        _finite_float_or_none(speed.get("acc_at_1")),
        _finite_float_or_none(course.get("acc_at_0.5")),
        _finite_float_or_none(course.get("acc_at_1")),
    ])
    scores: dict[str, float] = {}
    if text_cider is not None:
        scores["text_cider"] = text_cider
    if rmse_sum is not None:
        scores["control_rmse_negative"] = -float(rmse_sum)
    if threshold_mean is not None:
        scores["control_threshold_mean"] = threshold_mean
    if text_cider is not None:
        # Joint selection is text CIDEr plus continuous-control quality only.
        # Lower RMSE improves the score through the negative term.
        scores["adapt_joint"] = float(text_cider) + float(threshold_mean or 0.0) + 0.1 * float(scores.get("control_rmse_negative", 0.0))
    return scores


def update_best_selection_scores(
    previous_best: dict[str, float],
    scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, bool]]:
    best = dict(previous_best)
    improved: dict[str, bool] = {}
    for name, value in scores.items():
        if not math.isfinite(float(value)):
            improved[name] = False
            continue
        old = best.get(name)
        is_better = old is None or float(value) > float(old)
        improved[name] = bool(is_better)
        if is_better:
            best[name] = float(value)
    return best, improved


def _pearson_corr_or_none(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.detach().float().cpu().flatten()
    y = y.detach().float().cpu().flatten()
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.pow(2).sum() * y.pow(2).sum()).clamp_min(0.0))
    if float(denom.item()) <= 1e-12:
        return None
    return float((x * y).sum().div(denom).item())


def _pearson_corr_with_reason(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    delta_label: str,
) -> tuple[float | None, str]:
    x = x.detach().float().cpu().flatten()
    y = y.detach().float().cpu().flatten()
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 2:
        return None, "insufficient_valid_pairs"
    x = x[mask]
    y = y[mask]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_energy = x_centered.pow(2).sum()
    y_energy = y_centered.pow(2).sum()
    if float(x_energy.item()) <= 1e-12:
        return None, "factor_zero_variance"
    if float(y_energy.item()) <= 1e-12:
        return None, f"{delta_label}_zero_variance"
    denom = torch.sqrt((x_energy * y_energy).clamp_min(0.0))
    return float((x_centered * y_centered).sum().div(denom).item()), "ok"


def _delta_for_signal(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    signal_idx: int,
    signal_name: str | None = None,
) -> torch.Tensor:
    deltas = []
    for row in range(values.shape[0]):
        idxs = torch.nonzero(valid_mask[row, :, signal_idx], as_tuple=False).flatten()
        if idxs.numel() >= 2:
            delta = values[row, idxs[-1], signal_idx] - values[row, idxs[0], signal_idx]
            if signal_name is not None and is_circular_control_signal(signal_name):
                delta = signed_circular_delta_deg(delta)
            deltas.append(delta)
        else:
            deltas.append(torch.tensor(float("nan")))
    return torch.stack(deltas).float()


def _delta_distribution_stats(delta: torch.Tensor) -> dict[str, Any]:
    finite = delta[torch.isfinite(delta)]
    if finite.numel() == 0:
        return {
            "valid_count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "zero_variance": None,
        }
    std = float(finite.std(unbiased=False).item())
    return {
        "valid_count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": std,
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "zero_variance": bool(std <= 1e-12),
    }


def summarize_traffic_flow_audit(
    flow_probs: torch.Tensor | None,
    predicate_probs: torch.Tensor | None,
    pred_control: torch.Tensor,
    target_control: torch.Tensor,
    *,
    flow_factor_names: list[str] | None = None,
    predicate_names: list[str] | None = None,
    signal_names: list[str] | None = None,
    invalid_value: float = -1.0,
) -> dict[str, Any]:
    """Summarize whether macro traffic-flow factors align with control changes."""
    signal_names = list(signal_names or ["course", "speed"])
    pred = pred_control.detach().float().cpu()
    target = target_control.detach().float().cpu()
    valid = torch.isfinite(pred) & torch.isfinite(target) & target.ne(float(invalid_value))
    signal_deltas: dict[str, dict[str, torch.Tensor]] = {}
    for idx, name in enumerate(signal_names):
        signal_deltas[str(name)] = {
            "pred_delta": _delta_for_signal(pred, valid, idx, str(name)),
            "target_delta": _delta_for_signal(target, valid, idx, str(name)),
        }
    delta_stats: dict[str, Any] = {}
    for signal_name, deltas in signal_deltas.items():
        pred_stats = _delta_distribution_stats(deltas["pred_delta"])
        target_stats = _delta_distribution_stats(deltas["target_delta"])
        delta_stats[signal_name] = {
            "valid_pred_delta_count": pred_stats["valid_count"],
            "pred_delta_mean": pred_stats["mean"],
            "pred_delta_std": pred_stats["std"],
            "pred_delta_min": pred_stats["min"],
            "pred_delta_max": pred_stats["max"],
            "pred_delta_zero_variance": pred_stats["zero_variance"],
            "valid_target_delta_count": target_stats["valid_count"],
            "target_delta_mean": target_stats["mean"],
            "target_delta_std": target_stats["std"],
            "target_delta_min": target_stats["min"],
            "target_delta_max": target_stats["max"],
            "target_delta_zero_variance": target_stats["zero_variance"],
        }

    def stats_for_matrix(matrix: torch.Tensor | None, names: list[str] | None, key_prefix: str) -> dict[str, Any]:
        if matrix is None:
            return {"available": False}
        values = matrix.detach().float().cpu()
        if values.ndim == 3:
            values = values.mean(dim=1)
        if values.ndim != 2 or values.shape[0] == 0:
            return {"available": False, "reason": f"bad_{key_prefix}_shape", "shape": list(values.shape)}
        names_local = list(names or [f"{key_prefix}_{idx}" for idx in range(values.shape[1])])
        names_local = names_local[: values.shape[1]]
        per_factor: dict[str, Any] = {}
        top = []
        for idx, name in enumerate(names_local):
            series = values[:, idx]
            finite = series[torch.isfinite(series)]
            if finite.numel() == 0:
                mean = std = min_value = max_value = None
            else:
                mean = float(finite.mean().item())
                std = float(finite.std(unbiased=False).item())
                min_value = float(finite.min().item())
                max_value = float(finite.max().item())
            entry: dict[str, Any] = {
                "mean": mean,
                "std": std,
                "min": min_value,
                "max": max_value,
            }
            for signal_name, deltas in signal_deltas.items():
                pred_corr, pred_reason = _pearson_corr_with_reason(
                    series,
                    deltas["pred_delta"],
                    delta_label="pred_delta",
                )
                target_corr, target_reason = _pearson_corr_with_reason(
                    series,
                    deltas["target_delta"],
                    delta_label="target_delta",
                )
                entry[f"pred_{signal_name}_delta_corr"] = pred_corr
                entry[f"pred_{signal_name}_delta_corr_reason"] = pred_reason
                entry[f"target_{signal_name}_delta_corr"] = target_corr
                entry[f"target_{signal_name}_delta_corr_reason"] = target_reason
            if "speed" in signal_deltas:
                entry["target_speed_delta_corr"] = entry.get("target_speed_delta_corr")
            per_factor[str(name)] = entry
            if mean is not None:
                top.append({"name": str(name), "mean": mean, "std": std})
        top.sort(key=lambda item: float(item["mean"]), reverse=True)
        return {
            "available": True,
            "sample_count": int(values.shape[0]),
            "factor_count": int(values.shape[1]),
            "per_factor": per_factor,
            f"top_{key_prefix}_factors": top[:8],
        }

    flow_stats = stats_for_matrix(flow_probs, flow_factor_names, "flow")
    predicate_stats = stats_for_matrix(predicate_probs, predicate_names, "predicate")
    return {
        "available": bool(flow_stats.get("available") or predicate_stats.get("available")),
        "sample_count": int(pred.shape[0]),
        "signal_names": signal_names,
        "delta_stats": delta_stats,
        "flow_factors": flow_stats.get("per_factor", {}),
        "flow_factor_stats": flow_stats.get("per_factor", {}),
        "top_flow_factors": flow_stats.get("top_flow_factors", []),
        "predicate_factors": predicate_stats.get("per_factor", {}),
        "predicate_stats": predicate_stats.get("per_factor", {}),
        "top_predicate_factors": predicate_stats.get("top_predicate_factors", []),
        "notes": [
            "traffic-flow audit is diagnostic and excluded from checkpoint selection",
            "correlations are computed against continuous speed/course deltas, not discrete action proxies",
        ],
    }


def write_epoch_eval_artifacts(
    output_dir: str | Path,
    epoch_idx: int,
    eval_report: dict[str, Any],
    previous_best_metric: float | None,
) -> tuple[float, bool]:
    """Persist epoch eval and return updated best metric."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metric = float(eval_report["metric_value"])
    improved = previous_best_metric is None or metric > float(previous_best_metric)
    best_metric = metric if improved else float(previous_best_metric)
    record = dict(eval_report)
    record.update({"epoch": int(epoch_idx), "is_best": bool(improved), "best_metric_value": best_metric})
    (out / f"eval_epoch_{epoch_idx:03d}.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    with (out / "eval_summary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    return best_metric, improved


def _java_major_version(java_bin: str = "java") -> int | None:
    try:
        proc = subprocess.run([java_bin, "-version"], capture_output=True, text=True, check=False)
    except Exception:
        return None
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    match = re.search(r'version "([0-9]+)(?:\.([0-9]+))?', text)
    if not match:
        return None
    first = int(match.group(1))
    second = int(match.group(2) or 0)
    return second if first == 1 else first


def ensure_spice_java_compat_options(preferred_java_home: str | None = None, allow_java8_autodiscovery: bool = True) -> dict[str, Any]:
    """Make the legacy SPICE Java evaluator work on Java 8 or modern Java."""

    def activate_java_home(java_home: Path) -> None:
        java_bin_dir = java_home / "bin"
        os.environ["JAVA_HOME"] = str(java_home)
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(java_bin_dir) + os.pathsep + current_path

    if preferred_java_home:
        activate_java_home(Path(preferred_java_home))
    elif allow_java8_autodiscovery:
        current_major = _java_major_version("java")
        if current_major != 8:
            candidates = [
                Path(r"C:\Program Files\Eclipse Adoptium\jre-8.0.492.9-hotspot"),
                Path(r"C:\Program Files\Eclipse Adoptium"),
                Path(r"C:\Program Files\Java"),
                Path(r"C:\Program Files (x86)\Eclipse Adoptium"),
                Path(r"C:\Program Files (x86)\Java"),
            ]
            expanded: list[Path] = []
            for candidate in candidates:
                if candidate.is_dir() and (candidate / "bin" / "java.exe").exists():
                    expanded.append(candidate)
                elif candidate.is_dir():
                    expanded.extend(sorted(candidate.glob("*8*")))
            for candidate in expanded:
                java_exe = candidate / "bin" / "java.exe"
                if java_exe.exists() and _java_major_version(str(java_exe)) == 8:
                    activate_java_home(candidate)
                    break
    java_major = _java_major_version("java")
    current = os.environ.get("JAVA_TOOL_OPTIONS", "").strip()
    parts = [part for part in current.split() if not part.startswith("--add-opens=")]
    if java_major == 8:
        os.environ["JAVA_TOOL_OPTIONS"] = " ".join(parts).strip()
        return {"java_home": os.environ.get("JAVA_HOME"), "java_bin": "java", "java_major": java_major}
    required = [
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.math=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.text=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.time=ALL-UNNAMED",
    ]
    for option in required:
        if option not in parts:
            parts.append(option)
    os.environ["JAVA_TOOL_OPTIONS"] = " ".join(parts).strip()
    return {"java_home": os.environ.get("JAVA_HOME"), "java_bin": "java", "java_major": java_major}


def clear_spice_runtime_cache(repo_root: str | Path | None = None) -> None:
    """Clear SPICE LMDB/tmp state before ADAPT caption evaluation on Windows."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    spice_dir = root / "src" / "evalcap" / "coco_caption" / "pycocoevalcap" / "spice"
    for name in ("tmp", "cache"):
        target = spice_dir / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)


def parse_adapt_caption_eval_report(
    evaluate_file: str | Path,
    *,
    split: str,
    test_yaml: str,
    use_sep_cap: bool,
) -> dict[str, Any]:
    """Parse ADAPT COCO-caption eval files and expose the original selection metric."""
    evaluate_path = Path(evaluate_file)
    if use_sep_cap:
        des_path = Path(str(evaluate_path).replace("BDDX", "BDDX_des"))
        exp_path = Path(str(evaluate_path).replace("BDDX", "BDDX_exp"))
        if not des_path.exists() or not exp_path.exists():
            raise FileNotFoundError(f"Missing ADAPT sep-cap eval files: des={des_path.exists()} {des_path}, exp={exp_path.exists()} {exp_path}")
        des_metrics = json.loads(des_path.read_text(encoding="utf-8"))
        exp_metrics = json.loads(exp_path.read_text(encoding="utf-8"))
        metric_value = float(des_metrics["CIDEr"]) + float(exp_metrics["CIDEr"])
        return {
            "split": split,
            "adapt_aligned_test_yaml": test_yaml,
            "metric_family": "adapt_coco_caption",
            "metric_name": "CIDEr_des_plus_exp",
            "metric_value": metric_value,
            "higher_is_better": True,
            "evaluate_file": str(evaluate_path),
            "des_eval_file": str(des_path),
            "exp_eval_file": str(exp_path),
            "des_metrics": des_metrics,
            "exp_metrics": exp_metrics,
        }
    metrics = json.loads(evaluate_path.read_text(encoding="utf-8"))
    return {
        "split": split,
        "adapt_aligned_test_yaml": test_yaml,
        "metric_family": "adapt_coco_caption",
        "metric_name": "CIDEr",
        "metric_value": float(metrics["CIDEr"]),
        "higher_is_better": True,
        "evaluate_file": str(evaluate_path),
        "metrics": metrics,
    }


def _build_adapt_generation_eval_args(cfg: dict[str, Any], *, split: str, batch_size: int, beam_size: int, max_eval_samples: int = -1, device: str | torch.device = "cpu") -> SimpleNamespace:
    args = build_bddx_acpr_args(cfg, split=split, batch_size=batch_size, max_samples=max_eval_samples)
    args.device = str(device)
    args.val_yaml = cfg.get("paths", {}).get("test_yaml") if split == "test" else cfg.get("paths", {}).get("train_yaml")
    args.num_beams = int(beam_size)
    args.num_keep_best = 1
    args.num_return_sequences = 1
    args.temperature = 1.0
    args.top_k = 0
    args.top_p = 1.0
    args.repetition_penalty = 1.0
    args.length_penalty = 1.0
    args.output_hidden_states = False
    args.deepspeed_bf16 = False
    args.deepspeed_fp16 = False
    args.mixed_precision_method = ""
    args.use_cbs = False
    return args


def resolve_caption_tsv_from_dataset_yaml(dataset_yaml: str | Path) -> Path:
    import yaml

    yaml_path = Path(dataset_yaml)
    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    caption = spec.get("caption")
    if not caption:
        raise KeyError(f"Dataset yaml has no caption entry: {yaml_path}")
    caption_path = Path(caption)
    if not caption_path.is_absolute():
        caption_path = yaml_path.parent / caption_path
    return caption_path.resolve()


def evaluate_acpr_adapt_cider(
    model: ACPRFlowModel,
    cfg: dict[str, Any],
    *,
    split: str,
    batch_size: int,
    tokenizer,
    device: str | torch.device,
    beam_size: int,
    eval_output_dir: str | Path,
    max_eval_samples: int = -1,
) -> dict[str, Any]:
    """Run ADAPT generation eval on the configured split and return CIDEr metric report."""
    from src.tasks.run_adapt import evaluate as adapt_caption_evaluate

    eval_dir = Path(eval_output_dir)
    # ADAPT's TSV writer uses os.rename(tmp, final), which fails on Windows when
    # a previous eval artifact already exists. Epoch eval dirs are disposable
    # generation outputs, so clear them before each fresh evaluation.
    if eval_dir.exists():
        shutil.rmtree(eval_dir, ignore_errors=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    args = _build_adapt_generation_eval_args(
        cfg,
        split=split,
        batch_size=batch_size,
        beam_size=beam_size,
        max_eval_samples=max_eval_samples,
        device=device,
    )
    loader = build_bddx_acpr_dataloader(cfg, split=split, batch_size=batch_size, max_samples=max_eval_samples, tokenizer=tokenizer)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        ensure_spice_java_compat_options()
    os.environ.setdefault("SPICE_DISABLE_CACHE", "1")
    clear_spice_runtime_cache()
    evaluate_file = adapt_caption_evaluate(args, loader, model, tokenizer, str(eval_dir))
    if was_training:
        model.train()
    report = parse_adapt_caption_eval_report(
        evaluate_file,
        split=split,
        test_yaml=str(cfg.get("paths", {}).get("test_yaml")),
        use_sep_cap=bool(cfg.get("data", {}).get("use_sep_cap", True)),
    )
    dataset_yaml = cfg.get("paths", {}).get("test_yaml") if split == "test" else cfg.get("paths", {}).get("train_yaml")
    pred_tsv = Path(str(evaluate_file).replace(".eval.json", ".tsv"))
    if dataset_yaml and pred_tsv.exists():
        action_decision_report = evaluate_action_text_decisions(pred_tsv, resolve_caption_tsv_from_dataset_yaml(dataset_yaml))
        action_decision_report.update({
            "diagnostic_only": True,
            "excluded_from_checkpoint_selection": True,
            "reason": "text-derived action proxy is not an ADAPT primary metric",
        })
        (eval_dir / "diagnostic_action_text_decision_metrics.json").write_text(
            json.dumps(action_decision_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report["diagnostic_action_text_decision_metrics"] = action_decision_report
    report.update({
        "beam_size": int(beam_size),
        "batch_size": int(batch_size),
        "max_eval_samples": int(max_eval_samples),
        "eval_output_dir": str(eval_dir),
    })
    return report



def resolve_control_eval_yaml(cfg: dict[str, Any], split: str) -> str | None:
    """Return the ADAPT yaml that has continuous-control h5 coverage."""
    paths = cfg.get("paths", {})
    if split == "test":
        return paths.get("signal_test_yaml") or paths.get("test_yaml")
    return paths.get("signal_train_yaml") or paths.get("train_yaml")

def evaluate_acpr_control_metrics(
    model: ACPRFlowModel,
    cfg: dict[str, Any],
    *,
    split: str,
    batch_size: int,
    tokenizer,
    device: str | torch.device,
    eval_output_dir: str | Path,
    max_eval_samples: int = -1,
) -> dict[str, Any]:
    """Evaluate BDD-X continuous vehicle-control outputs on the configured split.

    This is intentionally separate from ADAPT caption CIDEr: CIDEr remains the
    paper-compatible text metric, while this report measures whether the model's
    control path predicts speed/course trajectories under ADAPT-style metrics.
    """
    eval_dir = Path(eval_output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    control_yaml = resolve_control_eval_yaml(cfg, split)
    control_cfg = dict(cfg)
    control_cfg["paths"] = dict(cfg.get("paths", {}))
    if control_yaml:
        control_cfg["paths"]["test_yaml" if split == "test" else "train_yaml"] = control_yaml
    loader = build_bddx_acpr_dataloader(
        control_cfg,
        split=split,
        batch_size=batch_size,
        max_samples=max_eval_samples,
        tokenizer=tokenizer,
    )
    precision = str(cfg.get("optimization", {}).get("precision", "fp32")).lower()
    use_bf16 = bool(str(device).startswith("cuda") and precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    was_training = model.training
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    sample_ids: list[str] = []
    flow_prob_batches: list[torch.Tensor] = []
    predicate_prob_batches: list[torch.Tensor] = []
    flow_factor_names: list[str] | None = None
    predicate_names: list[str] | None = None
    with torch.no_grad():
        for keys, examples, meta in loader:
            batch = adapt_batch_to_acpr_flow_batch(keys, examples, meta, device=device)
            if batch.car_info is None:
                continue
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=use_bf16):
                grids = model.backbone(batch.frames)
                bundle = model.build_bundle(batch.frames, precomputed_grids=grids)
                ctrl = model.predict_control_from_bundle(bundle, steps=batch.car_info.shape[-1])
                pred = ctrl["control_final_prediction"]
            preds.append(pred.detach().float().cpu())
            targets.append(batch.car_info.transpose(1, 2).contiguous().detach().float().cpu())
            sample_ids.extend(batch.sample_ids)
            flow_prob_batches.append(bundle.flow_probs.detach().float().cpu())
            predicate_prob_batches.append(bundle.predicate_probs_temporal.detach().float().mean(dim=1).cpu())
            if flow_factor_names is None:
                flow_factor_names = [str(x) for x in getattr(bundle, "flow_factor_names", [])]
                if not flow_factor_names and bundle.flow_probs.shape[-1] == len(FLOW_FACTOR_NAMES):
                    flow_factor_names = list(FLOW_FACTOR_NAMES)
            if predicate_names is None:
                predicate_names = [str(x) for x in getattr(bundle, "predicate_names", [])]
                if not predicate_names and bundle.predicate_probs_temporal.shape[-1] == len(ACPR_PREDICATE_NAMES):
                    predicate_names = list(ACPR_PREDICATE_NAMES)
    if not preds:
        report = {
            "available": False,
            "reason": "no_valid_control_signal_rows",
            "split": split,
            "max_eval_samples": int(max_eval_samples),
            "eval_output_dir": str(eval_dir),
        }
    else:
        pred_tensor = torch.cat(preds, dim=0)
        target_tensor = torch.cat(targets, dim=0)
        report = compute_control_metrics(
            pred_tensor,
            target_tensor,
            signal_names=list(cfg.get("data", {}).get("signals", ["course", "speed"])),
            invalid_value=-1.0,
            speed_delta_threshold=float(cfg.get("evaluation", {}).get("speed_delta_threshold", 0.5)),
            course_delta_threshold=float(cfg.get("evaluation", {}).get("course_delta_threshold", 0.05)),
            include_decision_proxy=False,
        )
        flow_tensor = torch.cat(flow_prob_batches, dim=0) if flow_prob_batches else None
        predicate_tensor = torch.cat(predicate_prob_batches, dim=0) if predicate_prob_batches else None
        traffic_flow_audit = summarize_traffic_flow_audit(
            flow_tensor,
            predicate_tensor,
            pred_tensor,
            target_tensor,
            flow_factor_names=flow_factor_names,
            predicate_names=predicate_names,
            signal_names=list(cfg.get("data", {}).get("signals", ["course", "speed"])),
            invalid_value=-1.0,
        )
        if int(report.get("valid_value_count", 0)) <= 0:
            report.update({
                "available": False,
                "reason": "no_valid_control_signal_rows",
            })
        else:
            report.update({
                "available": True,
                "metric_family": "adapt_continuous_control",
                "metric_name": "control_rmse_threshold_accuracy",
            })
        report.update({
            "split": split,
            "adapt_aligned_signal_yaml": str(control_yaml) if control_yaml else None,
            "max_eval_samples": int(max_eval_samples),
            "eval_output_dir": str(eval_dir),
            "sample_id_preview": sample_ids[:8],
            "traffic_flow_audit": traffic_flow_audit,
        })
        (eval_dir / "traffic_flow_audit.json").write_text(json.dumps(traffic_flow_audit, indent=2, sort_keys=True), encoding="utf-8")
    (eval_dir / "control_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if was_training:
        model.train()
    return report


def build_eval_safety_plan(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return epoch-eval safety controls with conservative defaults.

    Full ADAPT test CIDEr remains the only best-test selector. The smoke step is
    a cheap guard that catches generation/control eval failures before spending
    time on the full test split.
    """
    return {
        "save_pre_eval_checkpoint": bool(eval_cfg.get("save_pre_eval_checkpoint", True)),
        "smoke_before_full": bool(eval_cfg.get("smoke_before_full", False)),
        "smoke_max_eval_samples": int(eval_cfg.get("smoke_max_eval_samples", 32)),
        "control_smoke_max_eval_samples": int(eval_cfg.get("control_smoke_max_eval_samples", 64)),
    }


def assert_eval_smoke_report_is_healthy(report: dict[str, Any]) -> None:
    metric = float(report.get("metric_value", float("nan")))
    if not math.isfinite(metric):
        raise RuntimeError(f"ADAPT caption eval smoke produced a non-finite metric_value: {metric}")
    if "des_metrics" in report and "exp_metrics" in report:
        if not report["des_metrics"] or not report["exp_metrics"]:
            raise RuntimeError("ADAPT sep-cap eval smoke did not produce both description and explanation metrics")
    control_report = report.get("control_metrics", {})
    if control_report and control_report.get("available") is False:
        raise RuntimeError(f"Control eval smoke unavailable: {control_report.get('reason')}")


def run_sequence_calalign_stage(
    output_dir: str | Path,
    sample_ids: list[str],
    base_logits: torch.Tensor,
    enhanced_logits: torch.Tensor,
    targets: torch.Tensor,
) -> Path:
    """Fit Sequence-CalAlign on deterministic train-calib ids and write an audit artifact."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not all(str(s).startswith("train_calib_") for s in sample_ids):
        raise ValueError("Sequence-CalAlign stage may only fit deterministic train_calib sample ids")
    fitter = SequenceCalAlign(sample_ids)
    scales = fitter.fit(sample_ids, base_logits, enhanced_logits, targets)
    artifact = out / "sequence_calalign_audit.json"
    write_json(
        artifact,
        {
            **scales.__dict__,
            "fit_split": "train_calib",
            "fit_uses_test": fitter.fit_uses_test,
            "zero_alpha_candidate": True,
            "sample_count": len(sample_ids),
        },
    )
    return artifact


def train_formal(
    config: str,
    output_dir: str,
    device: str = "cpu",
    max_steps: int = 8,
    batch_size: int = 1,
    epochs: int = 1,
    beam_size: int = 1,
    gradient_accumulation_steps: int = 1,
    checkpoint_every_steps: int = 500,
    resume_checkpoint: str | None = None,
    load_pretrained_backbone: bool = True,
    max_eval_samples_override: int | None = None,
    num_workers_override: int | None = None,
    allow_windows_workers: bool = False,
) -> None:
    cfg = load_acpr_flow_config(config)
    if max_eval_samples_override is not None:
        cfg.setdefault("evaluation", {})["max_eval_samples"] = int(max_eval_samples_override)
    if num_workers_override is not None:
        cfg.setdefault("data", {})["num_workers"] = int(num_workers_override)
    if allow_windows_workers:
        cfg.setdefault("data", {})["allow_windows_workers"] = True
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    max_steps = int(max_steps)
    checkpoint_every_steps = int(checkpoint_every_steps)
    config_path = Path(config)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None
    run_manifest = {
        "config": str(config),
        "config_sha256": config_sha256,
        "output_dir": str(out),
        "device": str(device),
        "max_steps": max_steps,
        "max_steps_semantics": "0_or_negative_means_full_epoch_training_without_step_cap",
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": int(batch_size) * gradient_accumulation_steps,
        "epochs": int(epochs),
        "beam_size": int(beam_size),
        "checkpoint_every_steps": checkpoint_every_steps,
        "direct_image_training": bool(cfg.get("data", {}).get("direct_image_training", True)),
        "feature_cache_enabled": bool(cfg.get("data", {}).get("feature_cache_enabled", False)),
        "token_cache_enabled": bool(cfg.get("data", {}).get("token_cache_enabled", False)),
        "num_workers_effective_config": int(cfg.get("data", {}).get("num_workers", 4)),
        "allow_windows_workers": bool(cfg.get("data", {}).get("allow_windows_workers", False)),
        "max_eval_samples_override": int(max_eval_samples_override) if max_eval_samples_override is not None else None,
        "precision_requested": str(cfg.get("optimization", {}).get("precision", "fp32")),
        "started_at_unix": time.time(),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    suite = build_formal_experiment_suite(cfg)
    (out / "formal_suite_plan.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    checkpoint_report = load_formal_checkpoints(cfg)
    (out / "checkpoint_load_report.json").write_text(json.dumps(checkpoint_report, indent=2), encoding="utf-8")
    captioning_model = build_formal_captioning_model(cfg, device)
    checkpoint_report["adapt_trans_encoder_load_report"] = getattr(captioning_model, "acpr_adapt_trans_encoder_load_report", None)
    (out / "checkpoint_load_report.json").write_text(json.dumps(checkpoint_report, indent=2), encoding="utf-8")
    model = ACPRFlowModel(
        build_model_config(cfg, load_pretrained_backbone=load_pretrained_backbone),
        captioning_model=captioning_model,
    ).to(device)
    checkpoint_report["adapt_visual_path_load_report"] = load_adapt_visual_path_weights(model, cfg["paths"]["adapt_checkpoint"])
    (out / "checkpoint_load_report.json").write_text(json.dumps(checkpoint_report, indent=2), encoding="utf-8")
    groups, manifest = build_acpr_optimizer_groups(model)
    opt = torch.optim.AdamW(groups, weight_decay=float(cfg.get("optimization", {}).get("weight_decay", {}).get("new_modules", 0.01)))
    (out / "optimizer_group_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    resume_state = None
    start_epoch = 0
    global_step = 0
    optimizer_step = 0
    if resume_checkpoint:
        resume_state = load_resume_state(resume_checkpoint, model, opt, device=device)
        start_epoch = int(resume_state["start_epoch"])
        global_step = int(resume_state["global_step"])
        optimizer_step = int(resume_state["optimizer_step"])
        (out / "resume_state.json").write_text(json.dumps(resume_state, indent=2, sort_keys=True), encoding="utf-8")
    tokenizer_cls = _import_bert_tokenizer()
    tokenizer = tokenizer_cls.from_pretrained(cfg["paths"]["bert_dir"], do_lower_case=True)
    reason_embedding_weight = load_bert_word_embedding_weight(cfg["paths"]["bert_dir"], device=device)
    max_samples = max_steps * batch_size if max_steps > 0 else -1
    loader = build_bddx_acpr_dataloader(cfg, split="train", batch_size=batch_size, max_samples=max_samples, tokenizer=tokenizer)
    if resume_state is not None:
        raw_start_epoch = start_epoch
        start_epoch = resolve_resume_start_epoch(resume_state, len(loader))
        resume_state.update({"raw_start_epoch": raw_start_epoch, "resolved_start_epoch": start_epoch, "train_loader_len": len(loader)})
        (out / "resume_state.json").write_text(json.dumps(resume_state, indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = out / "metrics_summary.jsonl"
    metrics_file = metrics_path.open("w", encoding="utf-8")
    opt.zero_grad(set_to_none=True)
    precision = str(cfg.get("optimization", {}).get("precision", "fp32")).lower()
    use_bf16 = bool(str(device).startswith("cuda") and precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    run_manifest["precision_active"] = "bf16" if use_bf16 else "fp32"
    run_manifest["start_epoch"] = start_epoch
    run_manifest["initial_global_step"] = global_step
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    best_test_metric: float | None = None
    best_selection_scores: dict[str, float] = {}
    last_eval_report: dict[str, Any] | None = None

    try:
        for epoch in range(start_epoch, epochs):
            for keys, examples, meta in loader:
                batch = adapt_batch_to_acpr_flow_batch(keys, examples, meta, device=device)
                reason_semantic_target = build_reason_semantic_target_for_batch(
                    batch.raw_actions,
                    batch.raw_justifications,
                    tokenizer,
                    reason_embedding_weight,
                    device=device,
                    residual_strength=float(cfg.get("model", {}).get("reason_target", {}).get("action_residual_strength", 1.0)),
                )
                with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=use_bf16):
                    res = model(batch=batch, reason_semantic_target=reason_semantic_target)
                    loss_for_backward = res.total_loss / float(gradient_accumulation_steps)
                if not torch.isfinite(res.total_loss).all():
                    raise FloatingPointError(f"Non-finite total loss at global_step={global_step + 1}: {float(res.total_loss.detach().cpu())}")
                reason_norm = reason_semantic_target.norm(dim=-1).mean()
                if not torch.isfinite(reason_norm).all() or float(reason_norm.detach().cpu()) <= 0.0:
                    raise FloatingPointError(f"Invalid reason_semantic_target_norm at global_step={global_step + 1}: {float(reason_norm.detach().cpu())}")
                for name, value in res.loss_components.items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f"Non-finite loss component {name} at global_step={global_step + 1}")
                loss_for_backward.backward()
                global_step += 1
                rec = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "loss": float(res.total_loss.detach().cpu()),
                    "frames_shape": list(batch.frames.shape),
                    "sample_ids": batch.sample_ids[:2],
                    "reason_semantic_target_norm": float(reason_norm.detach().cpu()),
                    "loss_components": {k: float(v.detach().cpu()) for k, v in res.loss_components.items()},
                    "precision_active": run_manifest["precision_active"],
                }
                metrics_file.write(json.dumps(rec, ensure_ascii=True) + "\n")
                metrics_file.flush()
                print("ACPR_FLOW_BATCH " + json.dumps(rec, ensure_ascii=True), flush=True)
                should_step = (global_step % gradient_accumulation_steps) == 0
                if should_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("optimization", {}).get("gradient_clip_norm", 1.0)))
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(f"Non-finite gradient norm at global_step={global_step}: {float(grad_norm.detach().cpu())}")
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    optimizer_step += 1
                if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                    save_training_checkpoint(
                        out,
                        model,
                        opt,
                        epoch_idx=epoch,
                        global_step=global_step,
                        optimizer_step=optimizer_step,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        update_best=False,
                        eval_report=last_eval_report,
                    )
                if max_steps > 0 and global_step >= max_steps:
                    break
            if global_step % gradient_accumulation_steps != 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("optimization", {}).get("gradient_clip_norm", 1.0)))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at final accumulation step: {float(grad_norm.detach().cpu())}")
                opt.step()
                opt.zero_grad(set_to_none=True)
                optimizer_step += 1
            eval_cfg = cfg.get("evaluation", {})
            eval_safety = build_eval_safety_plan(eval_cfg)
            if eval_safety["save_pre_eval_checkpoint"]:
                save_training_checkpoint(
                    out,
                    model,
                    opt,
                    epoch_idx=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    update_best=False,
                    eval_report=last_eval_report,
                    epoch_complete=False,
                )
                print(
                    "ACPR_FLOW_PRE_EVAL_CHECKPOINT "
                    + json.dumps({"epoch": epoch, "global_step": global_step, "path": str(out / "checkpoint_latest.pth")}, ensure_ascii=True),
                    flush=True,
                )
            if eval_safety["smoke_before_full"]:
                smoke_report = evaluate_acpr_adapt_cider(
                    model,
                    cfg,
                    split=str(eval_cfg.get("best_selection_split", "test")),
                    batch_size=int(eval_cfg.get("eval_batch_size", batch_size)),
                    tokenizer=tokenizer,
                    device=device,
                    beam_size=int(eval_cfg.get("beam_size_smoke", eval_cfg.get("beam_size_formal", beam_size))),
                    eval_output_dir=out / f"adapt_eval_smoke_epoch_{epoch:03d}",
                    max_eval_samples=int(eval_safety["smoke_max_eval_samples"]),
                )
                smoke_control_report = evaluate_acpr_control_metrics(
                    model,
                    cfg,
                    split=str(eval_cfg.get("best_selection_split", "test")),
                    batch_size=int(eval_cfg.get("eval_batch_size", batch_size)),
                    tokenizer=tokenizer,
                    device=device,
                    eval_output_dir=out / f"control_eval_smoke_epoch_{epoch:03d}",
                    max_eval_samples=int(eval_safety["control_smoke_max_eval_samples"]),
                )
                smoke_report["control_metrics"] = smoke_control_report
                assert_eval_smoke_report_is_healthy(smoke_report)
                (out / f"eval_smoke_epoch_{epoch:03d}.json").write_text(
                    json.dumps(smoke_report, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print("ACPR_FLOW_EVAL_SMOKE " + json.dumps({"epoch": epoch, **smoke_report}, ensure_ascii=True), flush=True)
            eval_report = evaluate_acpr_adapt_cider(
                model,
                cfg,
                split=str(eval_cfg.get("best_selection_split", "test")),
                batch_size=int(eval_cfg.get("eval_batch_size", batch_size)),
                tokenizer=tokenizer,
                device=device,
                beam_size=int(eval_cfg.get("beam_size_formal", beam_size)),
                eval_output_dir=out / f"adapt_eval_epoch_{epoch:03d}",
                max_eval_samples=int(eval_cfg.get("max_eval_samples", -1)),
            )
            control_report = evaluate_acpr_control_metrics(
                model,
                cfg,
                split=str(eval_cfg.get("best_selection_split", "test")),
                batch_size=int(eval_cfg.get("eval_batch_size", batch_size)),
                tokenizer=tokenizer,
                device=device,
                eval_output_dir=out / f"control_eval_epoch_{epoch:03d}",
                max_eval_samples=int(eval_cfg.get("max_eval_samples", -1)),
            )
            eval_report["control_metrics"] = control_report
            if isinstance(control_report.get("traffic_flow_audit"), dict):
                (out / f"traffic_flow_audit_epoch_{epoch:03d}.json").write_text(
                    json.dumps(control_report["traffic_flow_audit"], indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            selection_scores = compute_adapt_selection_scores(eval_report)
            best_selection_scores, selection_improved = update_best_selection_scores(best_selection_scores, selection_scores)
            eval_report["selection_scores"] = selection_scores
            eval_report["selection_is_best"] = selection_improved
            eval_report["best_selection_scores"] = best_selection_scores
            last_eval_report = eval_report
            best_test_metric, improved = write_epoch_eval_artifacts(out, epoch, eval_report, best_test_metric)
            best_checkpoint_names: list[str] = []
            if selection_improved.get("text_cider", False):
                best_checkpoint_names.extend(["checkpoint_best_text.pth", "checkpoint_best_test.pth"])
            if selection_improved.get("control_rmse_negative", False):
                best_checkpoint_names.append("checkpoint_best_control.pth")
            if selection_improved.get("adapt_joint", False):
                best_checkpoint_names.append("checkpoint_best_adapt_joint.pth")
            save_training_checkpoint(
                out,
                model,
                opt,
                epoch_idx=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                gradient_accumulation_steps=gradient_accumulation_steps,
                update_best=bool(best_checkpoint_names),
                eval_report=eval_report,
                epoch_complete=True,
                best_checkpoint_names=best_checkpoint_names,
            )
            print(
                "ACPR_FLOW_EVAL "
                + json.dumps(
                    {
                        "epoch": epoch,
                        **eval_report,
                        "is_best": improved,
                        "best_checkpoint_names": best_checkpoint_names,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            if max_steps > 0 and global_step >= max_steps:
                break
        run_complete = {
            "completed": True,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "epochs_requested": int(epochs),
            "max_steps": max_steps,
            "finished_at_unix": time.time(),
            "metrics_summary": str(metrics_path),
            "best_test_metric": best_test_metric,
            "best_selection_scores": best_selection_scores,
            "last_eval_report": last_eval_report,
        }
        (out / "run_complete.json").write_text(json.dumps(run_complete, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        metrics_file.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--checkpoint_every_steps", type=int, default=500)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers_override", type=int, default=None)
    parser.add_argument("--allow_windows_workers", action="store_true")
    args = parser.parse_args()
    train_formal(
        args.config,
        args.output_dir,
        args.device,
        args.max_steps,
        args.batch_size,
        args.epochs,
        args.beam_size,
        args.gradient_accumulation_steps,
        args.checkpoint_every_steps,
        resume_checkpoint=args.resume_checkpoint,
        load_pretrained_backbone=not args.no_pretrained_backbone,
        max_eval_samples_override=args.max_eval_samples,
        num_workers_override=args.num_workers_override,
        allow_windows_workers=args.allow_windows_workers,
    )


if __name__ == "__main__":
    main()
