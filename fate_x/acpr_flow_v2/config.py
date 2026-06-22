from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class FlowCalV2Config:
    hidden_dim: int = 256
    state_dim: int = 256
    adapt_feature_dim: int = 512
    text_hidden_dim: int = 768
    text_vocab_size: int = 30522
    num_frames: int = 32
    image_resolution: int = 224
    num_predicates: int = 32
    num_semantic_factors: int = 13
    reason_memory_tokens: int = 54
    mask_prob: float = 0.5
    max_masked_tokens: int = 45
    direct_image_training: bool = True
    feature_cache_enabled: bool = False
    token_cache_enabled: bool = False
    use_real_video_swin: bool = False
    adapt_checkpoint: Optional[str] = None
    video_swin_checkpoint: Optional[str] = None
    bert_dir: str = "models/captioning/bert-base-uncased"
    epochs: int = 15
    stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "semantic_recovery",
            "epochs": [0, 1, 2],
            "train": [
                "transport",
                "predicate_tracker",
                "lane_flow_field",
                "axis_aware_flow",
                "reason_memory",
                "explanation_seca",
            ],
            "freeze": [
                "video_swin",
                "adapt_motion",
                "axis_control_adapter",
                "captioning_bert",
                "bert_layer_11",
                "lm_head_scst",
                "action_seca",
            ],
            "enable_control_losses": False,
        },
        {
            "name": "axis_aware_motion",
            "epochs": [3, 4, 5, 6, 7],
            "train": [
                "transport",
                "predicate_tracker",
                "lane_flow_field",
                "axis_aware_flow",
                "reason_memory",
                "axis_control_adapter",
            ],
            "freeze": [
                "video_swin",
                "adapt_motion",
                "captioning_bert",
                "bert_layer_11",
                "lm_head_scst",
                "temporal_seca",
            ],
            "enable_control_losses": True,
        },
        {
            "name": "conflict_aware_joint",
            "epochs": [8, 9, 10, 11, 12],
            "train": [
                "all_v2_modules",
                "axis_control_adapter",
                "adapt_motion_final_layer",
                "video_swin_final_stage",
                "bert_layer_11",
                "lm_head_scst",
            ],
            "freeze": [],
            "enable_control_losses": True,
        },
        {
            "name": "explanation_scst",
            "epochs": [13, 14],
            "train": [
                "explanation_seca",
                "lm_head_scst",
            ],
            "freeze": [
                "axis_control_adapter",
                "adapt_motion",
                "video_swin",
                "lane_flow_field",
            ],
            "enable_control_losses": False,
        },
    ])
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "action_text": 1.0,
        "explanation_text": 1.0,
        "speed_normalized": 1.0,
        "course_normalized": 1.0,
        "predicate_pu": 0.0,
        "flow_pu": 0.0,
        "reason_semantic": 0.0,
        "transport_consistency": 1.0,
        "lane_temporal_consistency": 0.0,
        "axis_direction_weak": 0.0,
    })
    optimization_learning_rates: Dict[str, float] = field(default_factory=lambda: {
        "new_modules": 1e-4,
    })
    optimization_weight_decay: Dict[str, float] = field(default_factory=lambda: {
        "new_modules": 0.01,
        "backbone": 0.05,
        "bias_norm_gate": 0.0,
    })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowCalV2Config":
        flat: Dict[str, Any] = {}
        data_section = data.get("data", {}) if isinstance(data, dict) else {}
        model_section = data.get("model", {}) if isinstance(data, dict) else {}
        training_section = data.get("training", {}) if isinstance(data, dict) else {}
        paths_section = data.get("paths", {}) if isinstance(data, dict) else {}
        if "state_dim" in model_section:
            flat["state_dim"] = model_section["state_dim"]
            flat["hidden_dim"] = model_section["state_dim"]
        if "text_hidden_dim" in model_section:
            flat["text_hidden_dim"] = model_section["text_hidden_dim"]
        if "adapt_feature_dim" in model_section:
            flat["adapt_feature_dim"] = model_section["adapt_feature_dim"]
        if "max_num_frames" in data_section:
            flat["num_frames"] = data_section["max_num_frames"]
        if "image_resolution" in data_section:
            flat["image_resolution"] = data_section["image_resolution"]
        backbone_section = model_section.get("backbone", {}) if isinstance(model_section, dict) else {}
        if backbone_section.get("type") == "adapt_video_swin_multiscale":
            flat["use_real_video_swin"] = True
        if "adapt_checkpoint" in paths_section:
            flat["adapt_checkpoint"] = paths_section["adapt_checkpoint"]
        if "video_swin_checkpoint" in paths_section:
            flat["video_swin_checkpoint"] = paths_section["video_swin_checkpoint"]
        if "bert_dir" in paths_section:
            flat["bert_dir"] = paths_section["bert_dir"]
        for key in ("direct_image_training", "feature_cache_enabled", "token_cache_enabled", "mask_prob", "max_masked_tokens"):
            if key in data_section:
                flat[key] = data_section[key]
        if "epochs" in training_section:
            flat["epochs"] = training_section["epochs"]
        if "stages" in data and isinstance(data["stages"], list):
            flat["stages"] = data["stages"]
        loss_section = data.get("loss", {}) if isinstance(data, dict) else {}
        if isinstance(loss_section, dict):
            defaults = cls().loss_weights
            defaults.update({str(k): float(v) for k, v in loss_section.items() if isinstance(v, (int, float))})
            flat["loss_weights"] = defaults
        optimization_section = data.get("optimization", {}) if isinstance(data, dict) else {}
        if isinstance(optimization_section, dict):
            lr_section = optimization_section.get("learning_rates", {})
            if isinstance(lr_section, dict):
                defaults = cls().optimization_learning_rates
                defaults.update({str(k): float(v) for k, v in lr_section.items() if isinstance(v, (int, float))})
                flat["optimization_learning_rates"] = defaults
            wd_section = optimization_section.get("weight_decay", {})
            if isinstance(wd_section, dict):
                defaults = cls().optimization_weight_decay
                defaults.update({str(k): float(v) for k, v in wd_section.items() if isinstance(v, (int, float))})
                flat["optimization_weight_decay"] = defaults
        cfg = cls(**flat)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.reason_memory_tokens != 54:
            raise ValueError("FlowCal V2 requires exactly 54 reason-memory tokens")
        if self.adapt_feature_dim != 512:
            raise ValueError("FlowCal V2 formal ADAPT dense tokens require adapt_feature_dim=512")
        if not self.direct_image_training or self.feature_cache_enabled or self.token_cache_enabled:
            raise ValueError("formal V2 requires direct images and no feature/token cache")
        seen = set()
        for stage in self.stages:
            for epoch in stage.get("epochs", []):
                if epoch in seen:
                    raise ValueError(f"overlapping stage epoch {epoch}")
                seen.add(epoch)
        if seen and (min(seen) != 0 or max(seen) != self.epochs - 1):
            raise ValueError("stage epochs must cover 0..epochs-1")

    def stage_for_epoch(self, epoch: int) -> str:
        for stage in self.stages:
            if epoch in stage.get("epochs", []):
                return stage.get("name", "unknown")
        raise KeyError(f"epoch {epoch} is not covered by any V2 stage")

    def stage_config(self, stage_name: str) -> Dict[str, Any]:
        for stage in self.stages:
            if stage.get("name") == stage_name:
                return stage
        raise KeyError(f"unknown V2 stage: {stage_name}")

    def stage_config_for_epoch(self, epoch: int) -> Dict[str, Any]:
        return self.stage_config(self.stage_for_epoch(epoch))

    def stage_enables_control_losses(self, stage_name: str) -> bool:
        stage = self.stage_config(stage_name)
        return bool(stage.get("enable_control_losses", False))

    def shared_text_loss_weight(self) -> float:
        # The current ADAPT-compatible masked language objective is shared by
        # description/action text and explanation text, so average the two plan
        # weights instead of accidentally doubling the only text loss tensor.
        return 0.5 * (float(self.loss_weights.get("action_text", 1.0)) + float(self.loss_weights.get("explanation_text", 1.0)))

    @property
    def text_contract(self) -> Dict[str, Any]:
        return {
            "mask_prob": self.mask_prob,
            "max_masked_tokens": self.max_masked_tokens,
            "max_seq_a_length": 15,
            "max_seq_length": 30,
            "use_sep_cap": True,
            "source": "flowcal_v2_config",
        }


def load_flowcal_v2_config(path: str | Path) -> FlowCalV2Config:
    path = Path(path)
    if yaml is None:
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return FlowCalV2Config.from_dict(data or {})


def write_resolved_config(config: FlowCalV2Config, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_config_binding_manifest(config), indent=2), encoding="utf-8")


def build_config_binding_manifest(config: FlowCalV2Config) -> Dict[str, Any]:
    config.validate()
    return {
        "config_class": "FlowCalV2Config",
        "direct_image_training": config.direct_image_training,
        "feature_cache_enabled": config.feature_cache_enabled,
        "token_cache_enabled": config.token_cache_enabled,
            "num_frames": config.num_frames,
            "image_resolution": config.image_resolution,
            "adapt_feature_dim": config.adapt_feature_dim,
            "reason_memory_tokens": config.reason_memory_tokens,
            "use_real_video_swin": config.use_real_video_swin,
            "adapt_checkpoint": config.adapt_checkpoint,
            "video_swin_checkpoint": config.video_swin_checkpoint,
            "mask_prob": config.mask_prob,
        "max_masked_tokens": config.max_masked_tokens,
        "stages": config.stages,
        "loss_weights": config.loss_weights,
        "optimization_learning_rates": config.optimization_learning_rates,
        "optimization_weight_decay": config.optimization_weight_decay,
    }


# Compatibility with earlier smoke tests.
ACPRFlowCalV2Config = FlowCalV2Config
load_v2_config = load_flowcal_v2_config
