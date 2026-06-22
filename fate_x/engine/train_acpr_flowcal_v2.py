from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

from fate_x.acpr_flow_v2.config import FlowCalV2Config, load_flowcal_v2_config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.engine.acpr_flowcal_v2_data import build_v2_dataloader
from fate_x.engine.eval_acpr_flowcal_v2 import evaluate as evaluate_v2_test
from fate_x.engine.train_acpr_flowcal_pp import build_formal_captioning_model

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

_LR_PREFIXES: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        ("transport", ("transport.",)),
        ("predicate_tracker", ("predicates.",)),
        ("lane_flow_field", ("lane_flow.",)),
        ("axis_aware_flow", ("flow.",)),
        ("reason_memory", ("memory.",)),
        ("action_seca", ("seca.query_action.", "seca.out_action.", "seca.gate_action")),
        ("explanation_seca", ("seca.query_explanation.", "seca.out_explanation.", "seca.gate_explanation")),
        ("temporal_seca", ("seca.",)),
        ("axis_control_adapter", ("control_adapter.",)),
        ("prefix_future", ("prefix_future.",)),
        ("hardpair_projection", ("hardpair.",)),
        ("bert_layer_11", ("captioning_model.bert.encoder.layer.11.", "captioning_bert.encoder.layer.11.", "bert.encoder.layer.11.")),
        ("captioning_bert", ("captioning_model.bert.",)),
        ("video_swin", ("video.", "video_swin.")),
        ("adapt_fc", ("video.fc.",)),
        ("adapt_motion", ("motion.", "adapt_motion.")),
        ("video_swin_final_stage", ("video.", "video_swin.")),
        ("adapt_motion_final_layer", ("motion.", "adapt_motion.")),
        ("adapt_motion_decoder", ("motion_decoder.", "adapt_motion_decoder.")),
        ("lm_head_scst", ("captioning_model.cls.", "lm_head.", "token_embed.")),
    ]
)


class StageController:
    def __init__(self, config: FlowCalV2Config):
        self.config = config

    def stage_for_epoch(self, epoch: int) -> str:
        return self.config.stage_for_epoch(epoch)

    def apply(self, model: nn.Module, epoch: int) -> Dict[str, Any]:
        stage = self.stage_for_epoch(epoch)
        train_prefixes = self._trainable_prefixes(stage)
        freeze_prefixes = self._freeze_prefixes(stage)
        for name, p in model.named_parameters():
            trainable = any(name.startswith(prefix) for prefix in train_prefixes)
            frozen = any(name.startswith(prefix) for prefix in freeze_prefixes)
            p.requires_grad = trainable and not frozen
        manifest = {"epoch": epoch, "stage": stage, "trainable": [n for n, p in model.named_parameters() if p.requires_grad]}
        return manifest

    def _trainable_prefixes(self, stage: str) -> tuple[str, ...]:
        stage_cfg = self.config.stage_config(stage)
        names = stage_cfg.get("train")
        if names:
            prefixes = self._prefixes_for_names(names)
            if prefixes:
                return prefixes
        return self._default_trainable_prefixes(stage)

    def _freeze_prefixes(self, stage: str) -> tuple[str, ...]:
        stage_cfg = self.config.stage_config(stage)
        return self._prefixes_for_names(stage_cfg.get("freeze") or [])

    def _prefixes_for_names(self, names: Iterable[str]) -> tuple[str, ...]:
        prefixes: list[str] = []
        for name in names:
            name = str(name)
            if name == "all_v2_modules":
                prefixes.extend(
                    prefix
                    for key, mapped in _LR_PREFIXES.items()
                    if key not in {"captioning_bert", "bert_layer_11", "video_swin", "adapt_fc", "adapt_motion", "video_swin_final_stage", "adapt_motion_final_layer", "adapt_motion_decoder", "lm_head_scst"}
                    for prefix in mapped
                )
                continue
            if name == "all_visual_state_modules":
                prefixes.extend(
                    prefix
                    for key in ("transport", "predicate_tracker", "lane_flow_field", "axis_aware_flow", "reason_memory")
                    for prefix in _LR_PREFIXES[key]
                )
                continue
            if name in {"lm_head", "lm_head_scst"}:
                prefixes.extend(_LR_PREFIXES["lm_head_scst"])
                continue
            mapped = _LR_PREFIXES.get(name)
            if mapped:
                prefixes.extend(mapped)
        return tuple(OrderedDict.fromkeys(prefixes))

    def _default_trainable_prefixes(self, stage: str) -> tuple[str, ...]:
        if stage == "semantic_recovery":
            return (
                "transport.",
                "predicates.",
                "lane_flow.",
                "flow.",
                "memory.",
                "seca.query_explanation.",
                "seca.out_explanation.",
                "seca.gate_explanation",
            )
        if stage == "axis_aware_motion":
            return ("transport.", "predicates.", "lane_flow.", "flow.", "memory.", "control_adapter.")
        if stage == "conflict_aware_joint":
            return ("video.", "transport.", "predicates.", "lane_flow.", "flow.", "memory.", "seca.", "control_adapter.", "motion.", "captioning_model.cls.", "lm_head.", "token_embed.")
        if stage == "explanation_scst":
            return ("seca.", "captioning_model.cls.", "lm_head.", "token_embed.")
        raise KeyError(f"unknown V2 stage: {stage}")

    def validate_trainable_manifest(self, manifest: Dict[str, Any]) -> bool:
        if "stage" not in manifest or "trainable" not in manifest:
            raise ValueError("invalid trainable manifest")
        return True


class StageAwareScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.05, min_lr_ratio: float = 0.10):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, int(self.total_steps * warmup_ratio))
        self.min_lr_ratio = min_lr_ratio
        self.step_count = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * scale

    def state_dict(self) -> Dict[str, Any]:
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_count = int(state.get("step_count", 0))
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))


class TestBestSelector:
    __test__ = False

    def __init__(self):
        self.best: Optional[tuple] = None

    def tuple_for_metrics(self, metrics: Dict[str, float]) -> tuple:
        control_violation = metrics.get("speed_rmse", 0.0) + metrics.get("course_rmse", 0.0)
        return (metrics.get("CIDEr_exp", 0.0), metrics.get("CIDEr_des", 0.0) + metrics.get("CIDEr_exp", 0.0), metrics.get("METEOR_exp", 0.0), -control_violation)

    def update(self, metrics: Dict[str, float]) -> bool:
        cur = self.tuple_for_metrics(metrics)
        if self.best is None or cur > self.best:
            self.best = cur
            return True
        return False


class _MetricSelector:
    def __init__(self, score_fn, eligible_fn=lambda metrics: True):
        self.score_fn = score_fn
        self.eligible_fn = eligible_fn
        self.best: Optional[tuple] = None

    def update(self, metrics: Dict[str, Any]) -> bool:
        if not self.eligible_fn(metrics):
            return False
        cur = self.score_fn(metrics)
        if self.best is None or cur > self.best:
            self.best = cur
            return True
        return False

    def seed(self, metrics: Dict[str, Any]) -> bool:
        if not self.eligible_fn(metrics):
            return False
        self.best = self.score_fn(metrics)
        return True


def _has_text_metrics(metrics: Dict[str, Any]) -> bool:
    return bool(metrics.get("text_metrics_available", False)) and "CIDEr_des" in metrics and "CIDEr_exp" in metrics


def _has_control_metrics(metrics: Dict[str, Any]) -> bool:
    return "speed_rmse" in metrics and "course_rmse" in metrics


def _control_tuple(metrics: Dict[str, Any]) -> tuple:
    rmse = float(metrics.get("speed_rmse", 1e9)) + float(metrics.get("course_rmse", 1e9))
    acc = float(metrics.get("speed_acc@1", 0.0)) + float(metrics.get("course_acc@1", 0.0))
    return (-rmse, acc)


def _text_tuple(metrics: Dict[str, Any]) -> tuple:
    return (float(metrics.get("CIDEr_des", 0.0)) + float(metrics.get("CIDEr_exp", 0.0)),)


def _explanation_tuple(metrics: Dict[str, Any]) -> tuple:
    return (float(metrics.get("CIDEr_exp", 0.0)), float(metrics.get("METEOR_exp", 0.0)))


def _joint_tuple(metrics: Dict[str, Any]) -> tuple:
    text_sum = float(metrics.get("CIDEr_des", 0.0)) + float(metrics.get("CIDEr_exp", 0.0))
    rmse = float(metrics.get("speed_rmse", 1e9)) + float(metrics.get("course_rmse", 1e9))
    acc = float(metrics.get("speed_acc@1", 0.0)) + float(metrics.get("course_acc@1", 0.0))
    return (text_sum, -rmse, acc)


def _test_tuple(metrics: Dict[str, Any]) -> tuple:
    control_violation = float(metrics.get("speed_rmse", 1e9)) + float(metrics.get("course_rmse", 1e9))
    return (
        float(metrics.get("CIDEr_exp", 0.0)),
        float(metrics.get("CIDEr_des", 0.0)) + float(metrics.get("CIDEr_exp", 0.0)),
        float(metrics.get("METEOR_exp", 0.0)),
        -control_violation,
    )


class BestCheckpointSuite:
    """Manage every V2 best-checkpoint contract without mixing illegal metrics."""

    def __init__(self, initial_metrics: Optional[Dict[str, Any]] = None):
        self.selectors = OrderedDict(
            [
                ("checkpoint_best_text.pth", _MetricSelector(_text_tuple, _has_text_metrics)),
                ("checkpoint_best_explanation.pth", _MetricSelector(_explanation_tuple, _has_text_metrics)),
                ("checkpoint_best_control.pth", _MetricSelector(_control_tuple, _has_control_metrics)),
                ("checkpoint_best_joint.pth", _MetricSelector(_joint_tuple, lambda m: _has_text_metrics(m) and _has_control_metrics(m))),
                ("checkpoint_best_test.pth", _MetricSelector(_test_tuple, lambda m: _has_text_metrics(m) and _has_control_metrics(m))),
            ]
        )
        if initial_metrics:
            self.seed(initial_metrics)

    def seed(self, metrics: Dict[str, Any]) -> list[str]:
        seeded: list[str] = []
        for filename, selector in self.selectors.items():
            if selector.seed(metrics):
                seeded.append(filename)
        return seeded

    def update_and_save(
        self,
        output_dir: str | Path,
        payload: Dict[str, Any],
        metrics: Dict[str, Any],
        baseline_text_floor: Optional[float] = None,
    ) -> list[str]:
        output_dir = Path(output_dir)
        updated: list[str] = []
        text_sum = _text_sum_value(metrics)
        text_floor_blocked = (
            baseline_text_floor is not None
            and text_sum is not None
            and float(text_sum) < float(baseline_text_floor)
        )
        text_floor_protected = {
            "checkpoint_best_text.pth",
            "checkpoint_best_explanation.pth",
            "checkpoint_best_joint.pth",
            "checkpoint_best_test.pth",
        }
        for filename, selector in self.selectors.items():
            if text_floor_blocked and filename in text_floor_protected:
                continue
            if selector.update(metrics):
                save_payload = dict(payload)
                save_payload["metrics"] = metrics
                save_payload["best_selector"] = filename
                if baseline_text_floor is not None:
                    save_payload["baseline_text_floor"] = float(baseline_text_floor)
                save_checkpoint_atomic(output_dir / filename, save_payload)
                updated.append(filename)
        return updated


def _append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _text_sum_value(metrics: Dict[str, Any]) -> Optional[float]:
    if not _has_text_metrics(metrics):
        return None
    return float(metrics.get("CIDEr_des", 0.0)) + float(metrics.get("CIDEr_exp", 0.0))


def assess_epoch_safety(
    metrics: Dict[str, Any],
    best_safe_text_sum: Optional[float],
    *,
    min_text_retention_ratio: float = 0.35,
    min_text_abs_drop: float = 0.05,
    max_text_negative_migration: float = 0.005,
) -> Dict[str, Any]:
    """Decide whether an evaluated epoch is safe to use for continuation.

    The trainer must not continue from an epoch that only improves a control
    proxy while destroying ADAPT text metrics. It must also not accept traffic
    flow audits where model-prediction correlations are missing.
    """
    reasons: list[str] = []
    text_sum = _text_sum_value(metrics)
    if text_sum is None:
        reasons.append("missing_adapt_text_metrics")
    elif best_safe_text_sum is not None and best_safe_text_sum > 0.0:
        abs_drop = best_safe_text_sum - text_sum
        retention = text_sum / best_safe_text_sum
        if abs_drop > max_text_negative_migration:
            reasons.append("adapt_text_metric_negative_migration")
        if abs_drop >= min_text_abs_drop and retention < min_text_retention_ratio:
            reasons.append("adapt_text_metric_collapse")

    for key in ("pred_speed_delta_corr", "pred_course_delta_corr"):
        if key in metrics and metrics.get(key) is None:
            reasons.append(f"missing_{key}")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "text_sum": text_sum,
        "best_safe_text_sum": best_safe_text_sum,
        "min_text_retention_ratio": min_text_retention_ratio,
        "min_text_abs_drop": min_text_abs_drop,
        "max_text_negative_migration": max_text_negative_migration,
    }


def write_eval_failure_record(
    output_dir: str | Path,
    *,
    epoch: int,
    checkpoint_path: str | Path,
    eval_output_dir: str | Path,
    exc: BaseException,
) -> Dict[str, Any]:
    record = {
        "epoch": epoch,
        "checkpoint_path": str(checkpoint_path),
        "eval_output_dir": str(eval_output_dir),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "action_required": "fix evaluation logic, then evaluate this checkpoint before resuming training",
        "resume_forbidden_until_epoch_eval": True,
        "created_at_unix": time.time(),
    }
    _append_jsonl(Path(output_dir) / "eval_failures.jsonl", record)
    return record


def write_epoch_rejection_record(
    output_dir: str | Path,
    *,
    epoch: int,
    checkpoint_path: str | Path,
    metrics: Dict[str, Any],
    safety: Dict[str, Any],
) -> Dict[str, Any]:
    record = {
        "epoch": epoch,
        "checkpoint_path": str(checkpoint_path),
        "safety": safety,
        "metrics": metrics,
        "action_required": "resume from the last safe best checkpoint; do not continue from checkpoint_latest for this rejected epoch",
        "created_at_unix": time.time(),
    }
    _append_jsonl(Path(output_dir) / "safety_rejections.jsonl", record)
    return record


class CheckpointMigratorV1ToV2:
    def migrate(self, checkpoint_path: Optional[str], model: nn.Module) -> Dict[str, Any]:
        if not checkpoint_path:
            return {"loaded": [], "missing": [], "unexpected": []}
        if checkpoint_path.endswith(".tmp"):
            raise ValueError(".tmp checkpoints are forbidden")
        if not Path(checkpoint_path).exists():
            return {"loaded": [], "missing": list(model.state_dict().keys()), "unexpected": []}
        state = torch.load(checkpoint_path, map_location="cpu")
        sd = state.get("model", state) if isinstance(state, dict) else {}
        result = model.load_state_dict(sd, strict=False)
        return {"loaded": list(sd.keys()), "missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}


def _lr_key_for_parameter(name: str) -> str:
    for key, prefixes in _LR_PREFIXES.items():
        if any(name.startswith(prefix) for prefix in prefixes):
            return key
    return "new_modules"


def _weight_decay_for_parameter(name: str, lr_key: str, config: FlowCalV2Config) -> float:
    wd = getattr(config, "optimization_weight_decay", {}) or {}
    if name.endswith("bias") or "norm" in name.lower() or "gate" in name.lower():
        return float(wd.get("bias_norm_gate", 0.0))
    if lr_key in {"video_swin_final_stage", "adapt_motion_final_layer", "adapt_motion_decoder", "bert_layer_11"}:
        return float(wd.get("backbone", wd.get("new_modules", 0.01)))
    return float(wd.get("new_modules", 0.01))


def build_optimizer_groups(model: nn.Module, config: FlowCalV2Config) -> list[dict]:
    grouped: "OrderedDict[tuple[str, float, float], list[torch.nn.Parameter]]" = OrderedDict()
    lrs = getattr(config, "optimization_learning_rates", {}) or {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_key = _lr_key_for_parameter(name)
        lr = float(lrs.get(lr_key, lrs.get("new_modules", 1e-4)))
        weight_decay = _weight_decay_for_parameter(name, lr_key, config)
        grouped.setdefault((lr_key, lr, weight_decay), []).append(param)
    return [
        {"params": params, "weight_decay": weight_decay, "lr": lr, "lr_key": lr_key}
        for (lr_key, lr, weight_decay), params in grouped.items()
    ]


def save_checkpoint_atomic(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp, _use_new_zipfile_serialization=False)
    os.replace(tmp, path)


def load_resume_exact(path: str | Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[StageAwareScheduler] = None) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix == ".tmp":
        raise ValueError(".tmp resume is forbidden")
    payload = torch.load(path, map_location="cpu")
    raw_state = payload["model"]
    target_state = model.state_dict()
    migrated_state, migration_meta = _migrate_historical_checkpoint_state(raw_state, target_state)
    incompatible = model.load_state_dict(migrated_state, strict=False)
    resume_meta: Dict[str, Any] = {}
    resume_meta["historical_key_migration"] = migration_meta
    resume_meta["missing_keys"] = list(getattr(incompatible, "missing_keys", []))
    resume_meta["unexpected_keys"] = list(getattr(incompatible, "unexpected_keys", []))
    if any(k.startswith("captioning_model.") for k in model.state_dict().keys()):
        loaded_caption_keys = [
            key for key in migrated_state.keys()
            if str(key).startswith("captioning_model.") or str(key).startswith("captioning_bert.") or str(key).startswith("bert.") or str(key).startswith("module.bert.")
        ]
        missing_caption = [key for key in resume_meta["missing_keys"] if str(key).startswith("captioning_model.")]
        if not loaded_caption_keys and missing_caption:
            raise RuntimeError(
                "resume checkpoint did not load ADAPT captioning_model weights; "
                f"missing_caption_keys={missing_caption[:20]}"
            )
        resume_meta["captioning_model_key_count"] = len(loaded_caption_keys)
    if optimizer is not None and payload.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(payload["optimizer"])
            resume_meta["optimizer_resume_loaded"] = True
        except ValueError as exc:
            resume_meta["optimizer_resume_loaded"] = False
            resume_meta["optimizer_resume_error"] = str(exc)
    elif optimizer is not None and "optimizer" in payload:
        resume_meta["optimizer_resume_loaded"] = False
        resume_meta["optimizer_resume_error"] = "checkpoint optimizer state is None"
    if scheduler is not None and payload.get("scheduler") is not None:
        state = payload["scheduler"]
        base_lrs = list(state.get("base_lrs", [])) if isinstance(state, dict) else []
        expected_groups = len(getattr(scheduler, "optimizer").param_groups)
        if base_lrs and len(base_lrs) != expected_groups:
            resume_meta["scheduler_resume_loaded"] = False
            resume_meta["scheduler_resume_error"] = f"scheduler base_lrs group count {len(base_lrs)} != optimizer group count {expected_groups}"
            if isinstance(state, dict) and "step_count" in state:
                scheduler.step_count = int(state["step_count"])
                resume_meta["scheduler_progress_fast_forwarded"] = True
                resume_meta["scheduler_progress_step_count"] = scheduler.step_count
        else:
            scheduler.load_state_dict(state)
            resume_meta["scheduler_resume_loaded"] = True
    elif scheduler is not None and "scheduler" in payload:
        resume_meta["scheduler_resume_loaded"] = False
        resume_meta["scheduler_resume_error"] = "checkpoint scheduler state is None"
    if resume_meta:
        payload = dict(payload)
        payload["resume_meta"] = {**payload.get("resume_meta", {}), **resume_meta}
    return payload


def _migrate_historical_checkpoint_state(source: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Map FlowCalPP/V1 historical checkpoint keys onto V2 names when shapes match."""
    migrated: Dict[str, torch.Tensor] = dict(source)
    rules = [
        ("backbone.video_swin.", "video.video_swin."),
        ("temporal_seca.q.", "seca.query_action."),
        ("temporal_seca.q.", "seca.query_explanation."),
        ("temporal_seca.out.", "seca.out_action."),
        ("temporal_seca.out.", "seca.out_explanation."),
        ("seca.query.", "seca.query_action."),
        ("seca.query.", "seca.query_explanation."),
        ("seca.out.", "seca.out_action."),
        ("seca.out.", "seca.out_explanation."),
        ("seca.gate", "seca.gate_action"),
        ("seca.gate", "seca.gate_explanation"),
        ("predicate_field.queries", "predicates.query"),
    ]
    loaded: list[dict[str, str]] = []
    shape_mismatch: list[dict[str, Any]] = []
    missing_target: list[dict[str, str]] = []
    for old_prefix, new_prefix in rules:
        for key, value in source.items():
            if key == old_prefix:
                new_key = new_prefix
            elif key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
            else:
                continue
            if new_key not in target:
                missing_target.append({"source": key, "target": new_key})
                continue
            if tuple(value.shape) != tuple(target[new_key].shape):
                shape_mismatch.append({"source": key, "target": new_key, "source_shape": list(value.shape), "target_shape": list(target[new_key].shape)})
                continue
            migrated[new_key] = value
            loaded.append({"source": key, "target": new_key})
    return migrated, {
        "rule_count": len(rules),
        "loaded_count": len(loaded),
        "shape_mismatch_count": len(shape_mismatch),
        "missing_target_count": len(missing_target),
        "loaded_preview": loaded[:30],
        "shape_mismatch_preview": shape_mismatch[:30],
        "missing_target_preview": missing_target[:30],
    }


def _load_raw_config_dict(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load V2 YAML config")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return data or {}


def _is_v2_checkpoint(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("config_version") == "acpr_flowcal_v2" or payload.get("trainer") == "train_acpr_flowcal_v2"


def _move_train_batch_to_device(batch: Any, device: str) -> Any:
    for attr in ("frames", "input_ids", "attention_mask", "token_type_ids", "masked_pos", "masked_ids", "car_info"):
        value = getattr(batch, attr, None)
        if isinstance(value, torch.Tensor):
            setattr(batch, attr, value.to(device))
    return batch


def _component_tensor(out: Any, name: str, fallback_attr: str) -> torch.Tensor:
    components = getattr(out, "loss_components", {}) or {}
    value = components.get(name)
    if value is None:
        value = getattr(out, fallback_attr)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(float(value), device=getattr(out, "total_loss").device)
    return value


def compute_stage_weighted_loss(
    out: Any,
    config: Optional[FlowCalV2Config],
    stage: Optional[str],
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Return the loss that should be backpropagated for the active V2 stage.

    The model exposes raw component losses for evaluation/auditing. The training
    contract lives in the YAML: early semantic recovery must not backpropagate
    control losses, while motion stages use the configured normalized control
    weights. Keeping this logic in the trainer prevents a model default from
    silently overriding the experiment plan.
    """
    if config is None or stage is None:
        return out.total_loss, {"raw_total": 1.0}

    control_enabled = config.stage_enables_control_losses(stage)
    weights = {
        "text": float(config.shared_text_loss_weight()),
        "speed": float(config.loss_weights.get("speed_normalized", 1.0)) if control_enabled else 0.0,
        "course": float(config.loss_weights.get("course_normalized", 1.0)) if control_enabled else 0.0,
        "auxiliary": 1.0,
        "control_enabled": 1.0 if control_enabled else 0.0,
    }
    text_loss = _component_tensor(out, "text", "action_text_loss")
    speed_loss = _component_tensor(out, "speed", "speed_loss")
    course_loss = _component_tensor(out, "course", "course_loss")
    auxiliary_loss = _component_tensor(out, "auxiliary", "auxiliary_loss")
    weighted = (
        weights["text"] * text_loss
        + weights["speed"] * speed_loss
        + weights["course"] * course_loss
        + weights["auxiliary"] * auxiliary_loss
    )
    if not torch.isfinite(weighted.detach()):
        raise FloatingPointError(f"non-finite weighted loss at stage={stage}: {float(weighted.detach().cpu())}")
    return weighted, weights


def train_one_epoch(
    model: ACPRFlowCalV2Model,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler: StageAwareScheduler,
    epoch: int,
    device: str = "cpu",
    gradient_accumulation_steps: int = 1,
    log_interval: int = 20,
    log_path: str | Path | None = None,
    config: Optional[FlowCalV2Config] = None,
    stage: Optional[str] = None,
) -> Dict[str, float]:
    model.train()
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    log_interval = max(1, int(log_interval))
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    losses = []
    optimizer_steps = 0
    start_time = time.time()
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    for step, batch in enumerate(loader):
        batch = _move_train_batch_to_device(batch, device)
        out = model(batch, stage=stage or "R")
        weighted_loss, loss_weights = compute_stage_weighted_loss(out, config, stage)
        (weighted_loss / gradient_accumulation_steps).backward()
        losses.append(float(weighted_loss.detach().cpu()))
        is_last = total_batches is not None and (step + 1) >= total_batches
        if (step + 1) % gradient_accumulation_steps == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
        if log_path is not None and (step == 0 or (step + 1) % log_interval == 0 or is_last):
            elapsed = max(1e-6, time.time() - start_time)
            done = step + 1
            eta_seconds = None
            if total_batches:
                eta_seconds = (elapsed / done) * max(0, total_batches - done)
            loss_components = getattr(out, "loss_components", {}) or {}
            row = {
                "epoch": epoch,
                "batch": done,
                "total_batches": total_batches,
                "stage": stage,
                "loss": float(weighted_loss.detach().cpu()),
                "weighted_loss": float(weighted_loss.detach().cpu()),
                "raw_total_loss": float(out.total_loss.detach().cpu()),
                "loss_mean_so_far": sum(losses) / max(1, len(losses)),
                "optimizer_steps": optimizer_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "eta_seconds": eta_seconds,
                "loss_components": {
                    str(k): float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                    for k, v in loss_components.items()
                },
                "loss_weights": loss_weights,
            }
            with Path(log_path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    return {"loss": sum(losses) / max(1, len(losses)), "steps": float(len(losses)), "optimizer_steps": float(optimizer_steps)}


@torch.no_grad()
def evaluate_after_epoch(
    model: ACPRFlowCalV2Model,
    loader: Iterable,
    epoch: int,
    device: str = "cpu",
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    metrics = evaluate_v2_test(
        model,
        loader,
        device=device,
        max_batches=-1,
        output_dir=output_dir,
        tokenizer=getattr(loader, "tokenizer", None),
    )
    metrics["eval_epoch"] = epoch
    metrics["adapt_text_metric_source"] = "adapt_sep_caption_eval" if metrics.get("text_metrics_available") else "blocked"
    return metrics


def _write_run_manifest(out_dir: Path, run_manifest: Dict[str, Any]) -> None:
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")


def _read_run_manifest(out_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(out_dir) / "run_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def recover_pending_eval_before_training(
    output_dir: str | Path,
    *,
    model: nn.Module,
    test_loader: Iterable,
    device: str,
    run_manifest: Dict[str, Any],
    checkpoint_suite: BestCheckpointSuite,
    best_safe_text_sum: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Evaluate a train-complete epoch that previously failed during eval.

    A training epoch is not usable until its checkpoint has been evaluated. If
    the previous process crashed in evaluation, the next launch must recover
    that exact checkpoint and stop before any further training. This prevents an
    epoch-1 training result from being silently skipped after fixing eval logic.
    """
    if not run_manifest.get("eval_failed"):
        return None
    checkpoint_path_raw = run_manifest.get("pending_eval_checkpoint")
    epoch_raw = run_manifest.get("pending_eval_epoch")
    if checkpoint_path_raw is None or epoch_raw is None:
        return None

    out_dir = Path(output_dir)
    checkpoint_path = Path(checkpoint_path_raw)
    epoch = int(epoch_raw)
    eval_output_dir = Path(run_manifest.get("pending_eval_output_dir") or out_dir / f"eval_epoch_{epoch:03d}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    model.to(device)

    eval_metrics = evaluate_after_epoch(model, test_loader, epoch, device=device, output_dir=eval_output_dir)
    safety = assess_epoch_safety(eval_metrics, best_safe_text_sum)
    recovered_payload = dict(payload) if isinstance(payload, dict) else {"model": model.state_dict(), "epoch": epoch}
    recovered_payload["metrics"] = eval_metrics
    recovered_payload["recovered_pending_eval"] = True
    recovered_payload["epoch"] = epoch

    if not safety["accepted"]:
        rejected_checkpoint_path = out_dir / f"checkpoint_rejected_epoch_{epoch:03d}.pth"
        recovered_payload["safety"] = safety
        save_checkpoint_atomic(rejected_checkpoint_path, recovered_payload)
        rejection_record = write_epoch_rejection_record(
            out_dir,
            epoch=epoch,
            checkpoint_path=rejected_checkpoint_path,
            metrics=eval_metrics,
            safety=safety,
        )
        row = {
            "epoch": epoch,
            **eval_metrics,
            "recovered_pending_eval": True,
            "safety_rejected": True,
            "safety_guard": safety,
            "best_checkpoint_updates": [],
        }
        _append_jsonl(out_dir / "metrics_summary.jsonl", row)
        run_manifest.update(
            {
                "eval_failed": False,
                "pending_eval_resolved": True,
                "pending_eval_resolved_at_unix": time.time(),
                "safety_rejected": True,
                "last_rejected_epoch": epoch,
                "last_rejection": rejection_record,
                "last_completed_epoch": epoch,
                "last_epoch_metrics": row,
                "history_rows": int(run_manifest.get("history_rows", 0)) + 1,
                "updated_at_unix": time.time(),
            }
        )
        _write_run_manifest(out_dir, run_manifest)
        return {"recovered": True, "safety_rejected": True, "epoch": epoch, "metrics": eval_metrics}

    best_updates = checkpoint_suite.update_and_save(
        out_dir,
        recovered_payload,
        eval_metrics,
        baseline_text_floor=best_safe_text_sum,
    )
    row = {
        "epoch": epoch,
        **eval_metrics,
        "recovered_pending_eval": True,
        "best_checkpoint_updates": best_updates,
    }
    _append_jsonl(out_dir / "metrics_summary.jsonl", row)
    run_manifest.update(
        {
            "eval_failed": False,
            "pending_eval_resolved": True,
            "pending_eval_resolved_at_unix": time.time(),
            "last_completed_epoch": epoch,
            "last_epoch_metrics": row,
            "history_rows": int(run_manifest.get("history_rows", 0)) + 1,
            "updated_at_unix": time.time(),
        }
    )
    run_manifest.pop("last_eval_failure", None)
    _write_run_manifest(out_dir, run_manifest)
    return {"recovered": True, "safety_rejected": False, "epoch": epoch, "metrics": eval_metrics, "best_checkpoint_updates": best_updates}


def run_formal_suite(
    config_path: str,
    output_dir: str,
    device: str = "cpu",
    epochs: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    gradient_accumulation_steps: int = 1,
    synthetic_smoke: bool = False,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
    eval_only: bool = False,
    eval_epoch: int = 0,
    baseline_text_sum: Optional[float] = None,
) -> Dict[str, Any]:
    cfg = load_flowcal_v2_config(config_path)
    if epochs is not None:
        cfg.epochs = epochs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cfg = _load_raw_config_dict(config_path) if Path(config_path).exists() else {}
    captioning_model = None if synthetic_smoke else build_formal_captioning_model(raw_cfg, device)
    model = ACPRFlowCalV2Model(cfg, captioning_model=captioning_model).to(device)
    controller = StageController(cfg)
    optimizer = torch.optim.AdamW(build_optimizer_groups(model, cfg))
    train_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "config_path": config_path,
        "formal": not synthetic_smoke,
        "synthetic": synthetic_smoke,
    }
    test_kwargs: Dict[str, Any] = dict(train_kwargs)
    if max_train_samples is not None:
        train_kwargs["length"] = max_train_samples
    if max_eval_samples is not None:
        test_kwargs["length"] = max_eval_samples
    train_loader = build_v2_dataloader("train", **train_kwargs)
    test_loader = build_v2_dataloader("test", **test_kwargs)
    scheduler = StageAwareScheduler(optimizer, total_steps=cfg.epochs * max(1, len(train_loader)))
    existing_manifest = _read_run_manifest(out_dir)
    start_epoch = 0
    resume_payload: Optional[Dict[str, Any]] = None
    resume_mode = "none"
    if resume_checkpoint:
        resume_payload = load_resume_exact(resume_checkpoint, model, optimizer=optimizer, scheduler=scheduler)
        if _is_v2_checkpoint(resume_payload):
            start_epoch = int(resume_payload.get("epoch", -1)) + 1
            resume_mode = "v2_resume"
        else:
            start_epoch = 0
            resume_mode = "historical_initialization"
    resume_metrics = resume_payload.get("metrics") if isinstance(resume_payload, dict) and isinstance(resume_payload.get("metrics"), dict) else None
    checkpoint_suite = BestCheckpointSuite(initial_metrics=resume_metrics)
    best_safe_text_sum = None
    if resume_metrics:
        best_safe_text_sum = _text_sum_value(resume_metrics)
    elif baseline_text_sum is not None:
        best_safe_text_sum = float(baseline_text_sum)
    elif existing_manifest and existing_manifest.get("best_safe_text_sum") is not None:
        best_safe_text_sum = float(existing_manifest["best_safe_text_sum"])
    history = []
    if existing_manifest and existing_manifest.get("eval_failed"):
        recovered = recover_pending_eval_before_training(
            out_dir,
            model=model,
            test_loader=test_loader,
            device=device,
            run_manifest=existing_manifest,
            checkpoint_suite=checkpoint_suite,
            best_safe_text_sum=best_safe_text_sum,
        )
        if recovered and recovered.get("recovered"):
            recovered_manifest = _read_run_manifest(out_dir) or existing_manifest
            return {"history": [recovered], "output_dir": str(out_dir), "run_manifest": recovered_manifest, "pending_eval_recovered": True}
    run_manifest = {
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "device": device,
        "epochs": cfg.epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "formal_eval_split": "test",
        "validation_loader_enabled": False,
        "synthetic_smoke": synthetic_smoke,
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "resume_checkpoint": resume_checkpoint,
        "start_epoch": start_epoch,
        "resume_mode": resume_mode,
        "resume_loaded": resume_payload is not None,
        "resume_meta": resume_payload.get("resume_meta") if isinstance(resume_payload, dict) else None,
        "captioning_model_enabled": captioning_model is not None,
        "captioning_model_load_report": getattr(captioning_model, "acpr_adapt_trans_encoder_load_report", None) if captioning_model is not None else None,
        "completed": False,
        "started_at_unix": time.time(),
        "last_completed_epoch": start_epoch - 1,
        "history_rows": 0,
        "resume_seeded_best_selectors": list(checkpoint_suite.selectors.keys()) if resume_metrics else [],
        "best_safe_text_sum": best_safe_text_sum,
        "train_progress_jsonl": str(out_dir / "train_progress.jsonl"),
        "metrics_summary_jsonl": str(out_dir / "metrics_summary.jsonl"),
    }
    _write_run_manifest(out_dir, run_manifest)
    if eval_only:
        eval_output_dir = out_dir / f"eval_epoch_{int(eval_epoch):03d}"
        eval_metrics = evaluate_after_epoch(model, test_loader, int(eval_epoch), device=device, output_dir=eval_output_dir)
        text_sum = _text_sum_value(eval_metrics)
        retention = None
        if baseline_text_sum is not None and baseline_text_sum > 0 and text_sum is not None:
            retention = float(text_sum) / float(baseline_text_sum)
        row = {
            "epoch": int(eval_epoch),
            **eval_metrics,
            "eval_only": True,
            "baseline_text_sum": baseline_text_sum,
            "baseline_text_retention": retention,
        }
        _append_jsonl(out_dir / "metrics_summary.jsonl", row)
        payload = {
            "model": model.state_dict(),
            "epoch": int(eval_epoch),
            "metrics": eval_metrics,
            "eval_only": True,
            "resume_checkpoint": resume_checkpoint,
            "config_version": "acpr_flowcal_v2",
            "trainer": "train_acpr_flowcal_v2",
        }
        best_updates = checkpoint_suite.update_and_save(
            out_dir,
            payload,
            eval_metrics,
            baseline_text_floor=best_safe_text_sum,
        )
        run_manifest.update(
            {
                "completed": True,
                "eval_only": True,
                "last_completed_epoch": int(eval_epoch),
                "last_epoch_metrics": row,
                "history_rows": 1,
                "baseline_text_sum": baseline_text_sum,
                "baseline_text_retention": retention,
                "best_checkpoint_updates": best_updates,
                "finished_at_unix": time.time(),
            }
        )
        _write_run_manifest(out_dir, run_manifest)
        return {"history": [row], "output_dir": str(out_dir), "run_manifest": run_manifest}
    for epoch in range(start_epoch, cfg.epochs):
        manifest = controller.apply(model, epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            epoch,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_interval=20,
            log_path=out_dir / "train_progress.jsonl",
            config=cfg,
            stage=manifest["stage"],
        )
        latest_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config_version": "acpr_flowcal_v2",
            "trainer": "train_acpr_flowcal_v2",
        }
        latest_checkpoint_path = out_dir / "checkpoint_latest.pth"
        eval_output_dir = out_dir / f"eval_epoch_{epoch:03d}"
        save_checkpoint_atomic(latest_checkpoint_path, latest_payload)
        try:
            eval_metrics = evaluate_after_epoch(model, test_loader, epoch, device=device, output_dir=eval_output_dir)
        except Exception as exc:
            failure_record = write_eval_failure_record(
                out_dir,
                epoch=epoch,
                checkpoint_path=latest_checkpoint_path,
                eval_output_dir=eval_output_dir,
                exc=exc,
            )
            run_manifest.update(
                {
                    "eval_failed": True,
                    "pending_eval_epoch": epoch,
                    "pending_eval_checkpoint": str(latest_checkpoint_path),
                    "pending_eval_output_dir": str(eval_output_dir),
                    "last_trained_epoch": epoch,
                    "last_eval_failure": failure_record,
                    "updated_at_unix": time.time(),
                }
            )
            _write_run_manifest(out_dir, run_manifest)
            raise RuntimeError(
                f"evaluation failed after epoch {epoch}; checkpoint was saved at {latest_checkpoint_path}. "
                "Fix evaluation and run this checkpoint before resuming training."
            ) from exc
        safety = assess_epoch_safety(eval_metrics, best_safe_text_sum)
        if not safety["accepted"]:
            rejected_payload = dict(latest_payload)
            rejected_payload["metrics"] = eval_metrics
            rejected_payload["safety"] = safety
            rejected_checkpoint_path = out_dir / f"checkpoint_rejected_epoch_{epoch:03d}.pth"
            save_checkpoint_atomic(rejected_checkpoint_path, rejected_payload)
            rejection_record = write_epoch_rejection_record(
                out_dir,
                epoch=epoch,
                checkpoint_path=rejected_checkpoint_path,
                metrics=eval_metrics,
                safety=safety,
            )
            row = {
                "epoch": epoch,
                **manifest,
                **train_metrics,
                **eval_metrics,
                "safety_rejected": True,
                "safety_guard": safety,
                "best_checkpoint_updates": [],
            }
            history.append(row)
            _append_jsonl(out_dir / "metrics_summary.jsonl", row)
            recommended_checkpoint = out_dir / "checkpoint_best_joint.pth"
            if not recommended_checkpoint.exists():
                recommended_checkpoint = Path(resume_checkpoint) if resume_checkpoint else out_dir / "checkpoint_latest.pth"
            run_manifest.update(
                {
                    "safety_rejected": True,
                    "last_rejected_epoch": epoch,
                    "last_rejection": rejection_record,
                    "resume_recommended_checkpoint": str(recommended_checkpoint),
                    "last_completed_epoch": epoch,
                    "history_rows": len(history),
                    "last_epoch_metrics": row,
                    "updated_at_unix": time.time(),
                }
            )
            _write_run_manifest(out_dir, run_manifest)
            raise RuntimeError(
                f"epoch {epoch} rejected by safety guard: {', '.join(safety['reasons'])}. "
                f"Resume from {recommended_checkpoint}, not from checkpoint_latest."
            )
        best_updates = checkpoint_suite.update_and_save(
            out_dir,
            latest_payload,
            eval_metrics,
            baseline_text_floor=best_safe_text_sum,
        )
        if safety["text_sum"] is not None:
            best_safe_text_sum = max(float(best_safe_text_sum or 0.0), float(safety["text_sum"]))
        row = {"epoch": epoch, **manifest, **train_metrics, **eval_metrics, "best_checkpoint_updates": best_updates}
        history.append(row)
        with (out_dir / "metrics_summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        run_manifest.update(
            {
                "last_completed_epoch": epoch,
                "history_rows": len(history),
                "last_epoch_metrics": row,
                "updated_at_unix": time.time(),
            }
        )
        _write_run_manifest(out_dir, run_manifest)
    run_manifest.update({"completed": True, "finished_at_unix": time.time(), "history_rows": len(history)})
    _write_run_manifest(out_dir, run_manifest)
    return {"history": history, "output_dir": str(out_dir), "run_manifest": run_manifest}


def train(args: argparse.Namespace) -> Dict[str, Any]:
    return run_formal_suite(
        args.config,
        args.output_dir,
        device=args.device,
        epochs=getattr(args, "epochs", None),
        batch_size=getattr(args, "batch_size", 1),
        num_workers=getattr(args, "num_workers", 0),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
        synthetic_smoke=getattr(args, "synthetic_smoke", False),
        max_train_samples=getattr(args, "max_train_samples", None),
        max_eval_samples=getattr(args, "max_eval_samples", None),
        resume_checkpoint=getattr(args, "resume", None),
        eval_only=getattr(args, "eval_only", False),
        eval_epoch=getattr(args, "eval_epoch", 0),
        baseline_text_sum=getattr(args, "baseline_text_sum", None),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--synthetic_smoke", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_epoch", type=int, default=0)
    parser.add_argument("--baseline_text_sum", type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
