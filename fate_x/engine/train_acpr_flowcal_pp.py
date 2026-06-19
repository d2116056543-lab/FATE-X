from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from fate_x.acpr_flow.model import ACPRFlowModel, ACPRFlowModelConfig
from fate_x.acpr_flow.online_reason_target import build_action_residual_reason_target
from fate_x.engine.acpr_bddx_data import (
    _import_bert_tokenizer,
    adapt_batch_to_acpr_flow_batch,
    assert_required_assets,
    build_bddx_acpr_dataloader,
)
from fate_x.utils.acpr_flow_config import load_acpr_flow_config


def build_acpr_optimizer_groups(model: ACPRFlowModel) -> tuple[list[dict], dict[str, str]]:
    lr_by_prefix = {
        "predicate_field": 1e-4,
        "flow_composer": 1e-4,
        "reason_memory": 1e-4,
        "temporal_seca": 5e-5,
        "reason_control_adapter": 2e-5,
        "prefix_future_head": 5e-5,
        "backbone": 5e-6,
        "control_base": 1e-5,
        "control_hidden": 1e-5,
    }
    groups: dict[float, dict] = {}
    manifest = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
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
    return ACPRFlowModelConfig(
        state_dim=int(model_cfg.get("state_dim", 256)),
        text_hidden_dim=int(model_cfg.get("text_hidden_dim", 768)),
        num_frames=int(cfg.get("data", {}).get("max_num_frames", 32)),
        image_resolution=int(cfg.get("data", {}).get("image_resolution", 224)),
        formal_backbone=True,
        load_pretrained_backbone=load_pretrained_backbone,
        video_swin_checkpoint=paths.get("video_swin_checkpoint"),
        fine_stage=int(multiscale.get("fine_stage", 2)),
        coarse_stage=int(multiscale.get("coarse_stage", 3)),
        use_transport=True,
        use_flow=True,
        use_prefix_future=True,
        invalid_control_value=float(cfg.get("data", {}).get("invalid_control_value", -1.0)),
    )


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


def train_formal(
    config: str,
    output_dir: str,
    device: str = "cpu",
    max_steps: int = 8,
    batch_size: int = 1,
    epochs: int = 1,
    load_pretrained_backbone: bool = True,
) -> None:
    cfg = load_acpr_flow_config(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite = build_formal_experiment_suite(cfg)
    (out / "formal_suite_plan.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    checkpoint_report = load_formal_checkpoints(cfg)
    (out / "checkpoint_load_report.json").write_text(json.dumps(checkpoint_report, indent=2), encoding="utf-8")
    model = ACPRFlowModel(build_model_config(cfg, load_pretrained_backbone=load_pretrained_backbone)).to(device)
    groups, manifest = build_acpr_optimizer_groups(model)
    opt = torch.optim.AdamW(groups, weight_decay=float(cfg.get("optimization", {}).get("weight_decay", {}).get("new_modules", 0.01)))
    (out / "optimizer_group_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tokenizer_cls = _import_bert_tokenizer()
    tokenizer = tokenizer_cls.from_pretrained(cfg["paths"]["bert_dir"], do_lower_case=True)
    reason_embedding_weight = load_bert_word_embedding_weight(cfg["paths"]["bert_dir"], device=device)
    loader = build_bddx_acpr_dataloader(cfg, split="train", batch_size=batch_size, max_samples=max_steps * batch_size, tokenizer=tokenizer)
    status = []
    global_step = 0
    for epoch in range(epochs):
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
            res = model(batch=batch, reason_semantic_target=reason_semantic_target)
            opt.zero_grad(set_to_none=True)
            res.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("optimization", {}).get("gradient_clip_norm", 1.0)))
            opt.step()
            global_step += 1
            rec = {
                "epoch": epoch,
                "global_step": global_step,
                "loss": float(res.total_loss.detach().cpu()),
                "frames_shape": list(batch.frames.shape),
                "sample_ids": batch.sample_ids[:2],
                "reason_semantic_target_norm": float(reason_semantic_target.norm(dim=-1).mean().detach().cpu()),
                "loss_components": {k: float(v.detach().cpu()) for k, v in res.loss_components.items()},
            }
            status.append(rec)
            print("ACPR_FLOW_BATCH " + json.dumps(rec, ensure_ascii=True), flush=True)
            if global_step >= max_steps:
                break
        torch.save({"model": model.state_dict(), "epoch": epoch, "global_step": global_step}, out / "checkpoint_latest.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch, "global_step": global_step}, out / "checkpoint_best_test.pth")
        if global_step >= max_steps:
            break
    (out / "metrics_summary.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=True) for x in status), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    args = parser.parse_args()
    train_formal(
        args.config,
        args.output_dir,
        args.device,
        args.max_steps,
        args.batch_size,
        args.epochs,
        load_pretrained_backbone=not args.no_pretrained_backbone,
    )


if __name__ == "__main__":
    main()
