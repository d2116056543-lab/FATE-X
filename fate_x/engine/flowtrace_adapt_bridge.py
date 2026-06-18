from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency error is environment-specific.
        raise RuntimeError("PyYAML is required to read FlowTrace config files.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"FlowTrace config must be a mapping: {path}")
    return data


def _as_cli_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


@dataclass(frozen=True)
class FlowTraceAdaptCommand:
    command: list[str]
    manifest: dict[str, Any]

    def write_manifest(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "flowtrace_adapt_command.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )


def build_adapt_train_command(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    epochs: int | None = None,
    micro_batch: int | None = None,
    grad_accum: int | None = None,
    max_steps: int | None = None,
    max_eval_samples: int | None = None,
    beam_size: int | None = None,
) -> FlowTraceAdaptCommand:
    """Build the real direct-image ADAPT training command required by FlowTrace.

    The command intentionally uses the public ADAPT dataloader with the plan's
    train/test yaml files. It does not create feature or token caches.
    """

    cfg_path = Path(config_path)
    cfg = _load_yaml(cfg_path)
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    baseline = cfg.get("baseline", {})
    model = cfg.get("model", {})
    opt = cfg.get("optimization", {})
    eval_cfg = cfg.get("evaluation", {})

    train_yaml = str(paths["train_yaml"]).replace("\\", "/")
    test_yaml = str(paths["test_yaml"]).replace("\\", "/")
    checkpoint_path = str(paths["adapt_checkpoint"]).replace("\\", "/")
    checkpoint_dir = str(Path(checkpoint_path).parent).replace("\\", "/") + "/"
    bert_dir = str(paths["bert_dir"]).replace("\\", "/")

    selected_epochs = int(epochs if epochs is not None else opt.get("epochs", 40))
    selected_batch = int(micro_batch if micro_batch is not None else 2)
    selected_accum = int(grad_accum if grad_accum is not None else 32)
    selected_beam = int(beam_size if beam_size is not None else eval_cfg.get("beam_size", 3))
    precision = str(opt.get("precision", "bf16")).lower()

    cmd = [
        sys.executable,
        "-u",
        "src/tasks/run_adapt.py",
        "--config",
        "src/configs/VidSwinBert/BDDX_multi_default.json",
        "--do_train",
        "--evaluate_during_training",
        "--data_dir",
        ".",
        "--train_yaml",
        train_yaml,
        "--val_yaml",
        test_yaml,
        "--model_name_or_path",
        bert_dir,
        "--pretrained_checkpoint",
        checkpoint_dir,
        "--output_dir",
        str(output_dir),
        "--num_train_epochs",
        str(selected_epochs),
        "--per_gpu_train_batch_size",
        str(selected_batch),
        "--per_gpu_eval_batch_size",
        str(max(selected_batch, 1)),
        "--gradient_accumulation_steps",
        str(selected_accum),
        "--learning_rate",
        str(opt.get("flowtrace_lr", 2.0e-4)),
        "--backbone_coef_lr",
        "0.05",
        "--max_num_frames",
        str(data.get("max_num_frames", 32)),
        "--img_res",
        str(data.get("image_resolution", 224)),
        "--pretrained_2d",
        "false",
        "--kinetics",
        "600",
        "--vidswin_size",
        "base",
        "--grid_feat",
        "true",
        "--mask_prob",
        "0.5",
        "--max_masked_tokens",
        "45",
        "--max_gen_length",
        str(baseline.get("action_max_tokens", 15)),
        "--num_beams",
        str(selected_beam),
        "--use_sep_cap",
        _as_cli_bool(baseline.get("use_sep_cap", True)),
        "--multitask",
        "true",
        "--signal_types",
        "course",
        "speed",
        "--loss_sensor_w",
        "0.05",
        "--max_grad_norm",
        str(opt.get("gradient_clip_norm", 1.0)),
        "--num_workers",
        str(data.get("num_workers", 4)),
        "--fate_x_enabled",
        "true",
        "--video_token_reducer",
        str(baseline.get("video_token_reducer", "none")),
        "--temporal_evidence_memory",
        str(baseline.get("temporal_evidence_memory", "none")),
        "--fate_x_reduce_control",
        _as_cli_bool(baseline.get("fate_x_reduce_control", False)),
        "--fate_x_text_reduce_only",
        "true",
        "--flowtrace_enabled",
        _as_cli_bool(model.get("flowtrace_enabled", True)),
        "--flowtrace_state_dim",
        str(model.get("state_dim", 256)),
        "--flowtrace_pmt_rank",
        str(model.get("token_pmt", {}).get("rank", 32)),
        "--learn_mask_enabled",
        "--attn_mask_type",
        "learn_vid_att",
        "--loss_sparse_w",
        "0.1",
    ]
    if precision == "bf16":
        cmd.extend([
            "--mixed_precision_method",
            "deepspeed",
            "--deepspeed_fp16",
            "false",
            "--deepspeed_bf16",
            "true",
            "--zero_opt_stage",
            str(opt.get("zero_opt_stage", 1)),
        ])
    elif precision == "fp16":
        cmd.extend([
            "--mixed_precision_method",
            "deepspeed",
            "--deepspeed_fp16",
            "true",
            "--deepspeed_bf16",
            "false",
            "--zero_opt_stage",
            str(opt.get("zero_opt_stage", 1)),
        ])
    if max_steps is not None and max_steps > 0:
        smoke_train_steps = max(1, int(max_steps))
        cmd.extend(["--limited_samples", str(smoke_train_steps * selected_batch)])
        cmd.extend(["--flowtrace_max_train_steps", str(smoke_train_steps)])
    if max_eval_samples is not None and max_eval_samples > 0:
        cmd.extend(["--limited_eval_samples", str(max(1, int(max_eval_samples)))])
        smoke_evidence = output_dir / "flowtrace_real_smoke_summary.json"
        cmd.extend(["--flowtrace_smoke_evidence", str(smoke_evidence).replace("\\", "/")])

    manifest = {
        "config_path": str(cfg_path),
        "output_dir": str(output_dir),
        "direct_image_training": bool(data.get("direct_image_training", False)),
        "feature_cache_enabled": bool(data.get("feature_cache_enabled", True)),
        "token_cache_enabled": bool(data.get("token_cache_enabled", True)),
        "train_yaml": train_yaml,
        "test_yaml_as_val_yaml": test_yaml,
        "adapt_checkpoint": checkpoint_path,
        "bert_dir": bert_dir,
        "epochs": selected_epochs,
        "micro_batch": selected_batch,
        "gradient_accumulation_steps": selected_accum,
        "effective_batch": selected_batch * selected_accum,
        "flowtrace_max_train_steps": max(1, int(max_steps)) if max_steps is not None and max_steps > 0 else None,
        "max_eval_samples": max_eval_samples,
        "flowtrace_smoke_evidence": str((output_dir / "flowtrace_real_smoke_summary.json")).replace("\\", "/")
        if max_eval_samples is not None and max_eval_samples > 0
        else None,
        "command": cmd,
    }
    return FlowTraceAdaptCommand(command=cmd, manifest=manifest)


def run_attached(command: FlowTraceAdaptCommand, output_dir: str | Path) -> int:
    out = Path(output_dir)
    command.write_manifest(out)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(command.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert proc.stdout is not None
    with (out / "train.log").open("a", encoding="utf-8") as log:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    return int(proc.wait())
