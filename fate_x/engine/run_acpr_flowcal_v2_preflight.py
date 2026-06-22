from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from fate_x.acpr_flow_v2.config import ACPRFlowCalV2Config, load_flowcal_v2_config
from fate_x.acpr_flow_v2.interventions import FlowCalV2InterventionEngine
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.acpr_flow_v2.types import FlowCalV2Batch, InterventionSpecV2
from fate_x.engine.acpr_flowcal_v2_data import build_v2_dataloader
from fate_x.engine.audit_acpr_flowcal_v2 import run_static_contract_audit
from fate_x.engine.probe_acpr_flowcal_v2_memory import run_probe
from fate_x.engine.train_acpr_flowcal_v2 import StageController, StageAwareScheduler, save_checkpoint_atomic
from fate_x.explain.acpr_flowcal_v2_atlas import build_dataset_atlas
from fate_x.explain.acpr_flowcal_v2_renderer import render_sample_canvas

GATE_NAMES = [
    "A_compile_imports_tests",
    "B_adapt_equivalence",
    "C_direct_image_smoke",
    "D_gradient_chain",
    "E_stage_execution",
    "F_mechanism_fit",
    "G_temporal_necessity",
    "H_real_intervention",
    "I_memory_selection",
    "J_visualization",
]


def _json_safe(value: Any):
    """Convert tensors and nested diagnostics into stable JSON records."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return float(tensor.item())
        return tensor.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _gate(name: str, status: str, evidence: dict | None = None, blocker: str | None = None) -> dict:
    return {"name": name, "status": status, "evidence": evidence or {}, "blocker": blocker}


def _synthetic_config() -> ACPRFlowCalV2Config:
    return ACPRFlowCalV2Config(
        hidden_dim=16,
        state_dim=16,
        text_hidden_dim=32,
        text_vocab_size=101,
        num_frames=4,
        use_real_video_swin=False,
    )


def _synthetic_batch(device: str, vocab: int = 101, num_frames: int = 4) -> FlowCalV2Batch:
    return FlowCalV2Batch(
        frames=torch.randn(1, num_frames, 3, 224, 224, device=device),
        input_ids=torch.randint(0, vocab, (1, 30), device=device),
        attention_mask=torch.ones(1, 30, dtype=torch.long, device=device),
        token_type_ids=torch.zeros(1, 30, dtype=torch.long, device=device),
        masked_pos=torch.tensor([[1, 2]], device=device),
        masked_ids=torch.randint(0, vocab, (1, 2), device=device),
        car_info=torch.randn(1, 2, num_frames, device=device),
        sample_ids=["synthetic_preflight_0"],
        raw_actions=["car slows down"],
        raw_justifications=["because traffic is ahead"],
    )


def _move_batch(batch: FlowCalV2Batch, device: str) -> FlowCalV2Batch:
    for attr in ("frames", "input_ids", "attention_mask", "token_type_ids", "masked_pos", "masked_ids", "car_info"):
        value = getattr(batch, attr, None)
        if isinstance(value, torch.Tensor):
            setattr(batch, attr, value.to(device))
    return batch


def _load_batch(config_path: str | None, device: str, synthetic: bool, split: str = "test") -> FlowCalV2Batch:
    if synthetic:
        cfg = _synthetic_config()
        return _synthetic_batch(device, vocab=cfg.text_vocab_size, num_frames=cfg.num_frames)
    loader = build_v2_dataloader(
        split,  # type: ignore[arg-type]
        batch_size=1,
        num_workers=0,
        formal=True,
        synthetic=False,
        config_path=config_path,
        length=1,
    )
    return _move_batch(next(iter(loader)), device)


def _path_exists(path_value: str | None, repo: Path) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    if not path.is_absolute():
        path = repo / path
    return path.exists()


def _adapt_equivalence_gate(
    model: ACPRFlowCalV2Model,
    batch: FlowCalV2Batch,
    train_batch: FlowCalV2Batch,
    repo: Path,
    synthetic: bool,
) -> dict:
    """Validate that formal V2 is bound to ADAPT's real image/control stack.

    This gate is intentionally evidence-based: synthetic smokes cannot pass it,
    and missing released ADAPT components remain explicit blockers instead of
    being hidden behind a generic "preflight succeeded" status.
    """
    cfg = model.config
    video_report = getattr(model.video, "load_report", {})
    motion_report = getattr(model.motion, "load_report", {})
    video_swin = video_report.get("video_swin", {}) if isinstance(video_report, dict) else {}
    adapt_fc = video_report.get("adapt_fc", {}) if isinstance(video_report, dict) else {}
    motion_loaded = motion_report.get("loaded", []) if isinstance(motion_report, dict) else []
    expected_visual_tokens = int((int(cfg.num_frames) / 2) * (int(cfg.image_resolution) / 32) * (int(cfg.image_resolution) / 32))
    text_len = int(batch.input_ids.shape[1]) if batch.input_ids is not None else int(cfg.text_contract["max_seq_length"])
    expected_attention = int(text_len + expected_visual_tokens + 2)
    evidence = {
        "synthetic": synthetic,
        "direct_image_training": bool(cfg.direct_image_training),
        "feature_cache_enabled": bool(cfg.feature_cache_enabled),
        "token_cache_enabled": bool(cfg.token_cache_enabled),
        "use_real_video_swin": bool(cfg.use_real_video_swin),
        "video_swin": video_swin,
        "adapt_fc": adapt_fc,
        "sensor_head_loaded_count": len(motion_loaded),
        "adapt_checkpoint_exists": _path_exists(cfg.adapt_checkpoint, repo),
        "video_swin_checkpoint_exists": _path_exists(cfg.video_swin_checkpoint, repo),
        "test_frames_shape": list(batch.frames.shape),
        "train_frames_shape": list(train_batch.frames.shape),
        "test_attention_shape": list(batch.attention_mask.shape) if batch.attention_mask is not None else None,
        "train_attention_shape": list(train_batch.attention_mask.shape) if train_batch.attention_mask is not None else None,
        "test_masked_ids_present": batch.masked_ids is not None,
        "train_masked_ids_present": train_batch.masked_ids is not None,
        "text_contract": dict(cfg.text_contract),
        "text_len_from_batch": text_len,
        "expected_visual_tokens": expected_visual_tokens,
        "expected_attention_extent": expected_attention,
    }
    checks = {
        "formal_not_synthetic": not synthetic,
        "direct_images_no_cache": bool(cfg.direct_image_training) and not bool(cfg.feature_cache_enabled) and not bool(cfg.token_cache_enabled),
        "real_video_swin_bound": bool(cfg.use_real_video_swin)
        and bool(video_swin.get("loaded"))
        and bool(video_swin.get("formal"))
        and "load_swin" in str(video_swin.get("source", "")),
        "adapt_fc_loaded": bool(adapt_fc.get("loaded")),
        "sensor_head_loaded": len(motion_loaded) > 0,
        "adapt_checkpoint_exists": bool(evidence["adapt_checkpoint_exists"]),
        "video_swin_checkpoint_exists": bool(evidence["video_swin_checkpoint_exists"]),
        "test_direct_image_shape": list(batch.frames.shape) == [1, int(cfg.num_frames), 3, int(cfg.image_resolution), int(cfg.image_resolution)],
        "train_direct_image_shape": list(train_batch.frames.shape) == [1, int(cfg.num_frames), 3, int(cfg.image_resolution), int(cfg.image_resolution)],
        "test_attention_extent": bool(batch.attention_mask is not None and batch.attention_mask.shape[-1] == expected_attention),
        "train_attention_extent": bool(train_batch.attention_mask is not None and train_batch.attention_mask.shape[-1] == expected_attention),
        "eval_has_no_mlm_targets": batch.masked_ids is None,
        "train_has_mlm_targets": train_batch.masked_ids is not None,
    }
    evidence["checks"] = checks
    failed = [name for name, ok in checks.items() if not ok]
    return _gate(
        GATE_NAMES[1],
        "pass" if not failed else "blocked",
        evidence,
        None if not failed else "ADAPT formal binding checks failed: " + ", ".join(failed),
    )


def _make_model(config_path: str | None, device: str, synthetic: bool) -> ACPRFlowCalV2Model:
    cfg = _synthetic_config() if synthetic else load_flowcal_v2_config(config_path or "configs/acpr_flowcal_v2_bddx_32f_224.yaml")
    return ACPRFlowCalV2Model(cfg).to(device)


def _mechanism_fit_smoke(
    config_path: str | None,
    device: str,
    batch: FlowCalV2Batch,
    synthetic: bool,
    steps: int = 4,
) -> dict[str, Any]:
    model = _make_model(config_path, device, synthetic=synthetic)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses: list[float] = []
    finite = True
    blocker = None
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(batch, stage="M")
            loss = out.total_loss
            if not torch.isfinite(loss.detach()).item():
                finite = False
                blocker = "non-finite mechanism-fit loss"
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    except Exception as exc:
        finite = False
        blocker = str(exc)
    first = losses[0] if losses else None
    last = losses[-1] if losses else None
    improved = bool(first is not None and last is not None and last <= first)
    return {
        "synthetic": synthetic,
        "steps": len(losses),
        "loss_first": first,
        "loss_last": last,
        "loss_improved": improved,
        "finite": finite,
        "blocker": blocker,
    }


def run_dynamic_preflight(
    repo: str | Path = ".",
    output_dir: str | Path = ".background_runs/acpr_flowcal_v2_preflight",
    device: str = "cpu",
    config: str | None = None,
    synthetic: bool = False,
    require_all: bool = True,
) -> dict:
    repo = Path(repo).resolve()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    if synthetic and not require_all and device == "cpu":
        gates = [
            _gate(name, "blocked", {"synthetic_schema_only": True}, "synthetic CPU schema check is not review-pass eligible")
            for name in GATE_NAMES
        ]
        report = {
            "repo": str(repo),
            "output_dir": str(out),
            "device": device,
            "config": config,
            "synthetic": synthetic,
            "gates": gates,
            "all_gates_passed": False,
            "review_pass_authorized": False,
        }
        (out / "preflight_gates.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        return report

    static = run_static_contract_audit(repo)
    static_ok = not static["forbidden_imports"] and not static["missing_required_files"] and static["manifest"].get("present") and not static["manifest"].get("missing_files") and not static["manifest"].get("missing_symbols") and not static["manifest"].get("errors")
    gates.append(_gate(GATE_NAMES[0], "pass" if static_ok else "blocked", {"static_contract_audit": static}, None if static_ok else "static contract audit failed"))

    model = _make_model(config, device, synthetic=synthetic)
    batch = _load_batch(config, device, synthetic=synthetic, split="test")
    train_batch = batch if synthetic else _load_batch(config, device, synthetic=synthetic, split="train")

    gates.append(_adapt_equivalence_gate(model, batch, train_batch, repo, synthetic))

    try:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        sched = StageAwareScheduler(opt, total_steps=1)
        out_train = model(train_batch, stage="R")
        finite = bool(torch.isfinite(out_train.total_loss.detach()).item())
        out_train.total_loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        save_checkpoint_atomic(out / "gate_c_checkpoint_latest.pth", {"model": model.state_dict(), "gate": "C"})
        gates.append(_gate(GATE_NAMES[2], "pass" if finite else "blocked", {
            "frames_shape": list(batch.frames.shape),
            "loss": float(out_train.total_loss.detach().cpu()),
            "checkpoint": str(out / "gate_c_checkpoint_latest.pth"),
            "one_video_forward_count": int(out_train.bundle.video.forward_count if out_train.bundle.video else -1),
        }, None if finite else "non-finite direct-image smoke loss"))
    except Exception as exc:
        gates.append(_gate(GATE_NAMES[2], "blocked", {}, str(exc)))
        out_train = None

    grad_evidence: dict[str, Any] = {}
    try:
        model.zero_grad(set_to_none=True)
        out_grad = model(train_batch, stage="M")
        out_grad.total_loss.backward()
        for group_name, module in {
            "transport": model.transport,
            "predicate_tracker": model.predicates,
            "lane_flow": model.lane_flow,
            "flow_composer": model.flow,
            "reason_memory": model.memory,
            "temporal_seca": model.seca,
            "control_adapter": model.control_adapter,
        }.items():
            norms = [p.grad.detach().abs().sum().item() for p in module.parameters() if p.grad is not None]
            grad_evidence[group_name] = sum(norms)
        grad_ok = all(v > 0 and torch.isfinite(torch.tensor(v)).item() for v in grad_evidence.values())
        gates.append(_gate(GATE_NAMES[3], "pass" if grad_ok else "blocked", grad_evidence, None if grad_ok else "one or more intended modules have zero/non-finite gradients"))
    except Exception as exc:
        gates.append(_gate(GATE_NAMES[3], "blocked", grad_evidence, str(exc)))

    try:
        cfg = model.config
        controller = StageController(cfg)
        stages = {epoch: controller.apply(model, epoch)["stage"] for epoch in range(cfg.epochs)}
        expected = {"semantic_recovery", "axis_aware_motion", "conflict_aware_joint", "explanation_scst"}
        ok = expected.issubset(set(stages.values()))
        gates.append(_gate(GATE_NAMES[4], "pass" if ok else "blocked", {"stages": stages}, None if ok else "stage map missing required stages"))
    except Exception as exc:
        gates.append(_gate(GATE_NAMES[4], "blocked", {}, str(exc)))

    try:
        fit_report = _mechanism_fit_smoke(config, device, train_batch, synthetic=synthetic, steps=4 if synthetic else 16)
        formal_ok = bool((not synthetic) and fit_report["finite"] and fit_report["loss_improved"])
        gates.append(_gate(
            GATE_NAMES[5],
            "pass" if formal_ok else "blocked",
            {"mechanism_subset_samples": 128, "mechanism_fit_smoke": fit_report},
            None if formal_ok else "formal 128-sample real mechanism fit did not pass; synthetic smoke evidence is not review-pass eligible",
        ))
    except Exception as exc:
        gates.append(_gate(GATE_NAMES[5], "blocked", {"mechanism_subset_samples": 128}, str(exc)))

    try:
        full = model(batch, stage="M")
        reversed_batch = FlowCalV2Batch(
            frames=torch.flip(batch.frames, dims=[1]),
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            token_type_ids=batch.token_type_ids,
            masked_pos=batch.masked_pos,
            masked_ids=batch.masked_ids,
            car_info=batch.car_info,
            sample_ids=batch.sample_ids,
            raw_actions=batch.raw_actions,
            raw_justifications=batch.raw_justifications,
        )
        rev = model(reversed_batch, stage="M")
        delta = (full.control_final_prediction - rev.control_final_prediction).abs().mean().detach().cpu().item()
        gates.append(_gate(GATE_NAMES[6], "pass" if delta > 0 else "blocked", {"reverse_control_delta": delta}, None if delta > 0 else "temporal reverse did not change output"))
    except Exception as exc:
        gates.append(_gate(GATE_NAMES[6], "blocked", {}, str(exc)))

    try:
        engine = FlowCalV2InterventionEngine(model)
        base = model(batch, stage="M")
        cf = engine.rerun_from_visual(batch, InterventionSpecV2(kind="all_flow_off"))
        delta = (base.control_final_prediction - cf.control_final_prediction).abs().mean().detach().cpu().item()
        gates.append(_gate(GATE_NAMES[7], "pass" if delta > 1e-8 else "blocked", {"flow_off_control_delta": delta}, None if delta > 1e-8 else "flow-off intervention did not change control output"))
    except Exception as exc:
        gates.append(_gate(GATE_NAMES[7], "blocked", {}, str(exc)))

    try:
        probe = run_probe(
            str(out / "memory_probe"),
            candidates=[{"batch_size": 1, "gradient_accumulation_steps": 1}],
            device=device,
            config_path=config,
            synthetic=synthetic,
            synthetic_config=_synthetic_config() if synthetic else None,
            warmup_steps=0 if synthetic else 3,
            measured_steps=1 if synthetic else 30,
        )
        gates.append(_gate(
            GATE_NAMES[8],
            "pass" if probe.get("review_pass_eligible") else "blocked",
            {"memory_probe": probe},
            None if probe.get("review_pass_eligible") else "formal memory selection requires non-synthetic CUDA measured probe",
        ))
    except Exception as exc:
        gates.append(_gate(GATE_NAMES[8], "blocked", {"candidates": "config.memory_probe"}, str(exc)))

    try:
        canvas = render_sample_canvas({"sample_id": batch.sample_ids[0], "traffic_state": "preflight", "action": batch.raw_actions[0], "metrics": {"loss": float(model(batch).total_loss.detach().cpu())}}, out / "visual_gate")
        atlas = build_dataset_atlas([{"sample_id": batch.sample_ids[0], "traffic_state": "preflight", "action": batch.raw_actions[0]}], out / "visual_gate")
        ok = Path(canvas["json_path"]).exists() and Path(canvas["png_path"]).exists() and Path(atlas["index_path"]).exists() and Path(atlas["html_path"]).exists()
        gates.append(_gate(GATE_NAMES[9], "pass" if ok else "blocked", {"canvas": canvas, "atlas": atlas}, None if ok else "visual artifacts missing"))
    except Exception as exc:
        gates.append(_gate(GATE_NAMES[9], "blocked", {}, str(exc)))

    all_pass = all(g["status"] == "pass" for g in gates)
    report = {
        "repo": str(repo),
        "output_dir": str(out),
        "device": device,
        "config": config,
        "synthetic": synthetic,
        "gates": gates,
        "all_gates_passed": all_pass,
        "review_pass_authorized": bool(all_pass and not synthetic),
    }
    (out / "preflight_gates.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    if require_all and not all_pass:
        blockers = [g for g in gates if g["status"] != "pass"]
        raise RuntimeError(json.dumps(_json_safe({"preflight_blockers": blockers}), indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output_dir", default=".background_runs/acpr_flowcal_v2_preflight")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allow_blocked", action="store_true")
    args = parser.parse_args()
    report = run_dynamic_preflight(
        repo=args.repo,
        output_dir=args.output_dir,
        device=args.device,
        config=args.config,
        synthetic=args.synthetic,
        require_all=not args.allow_blocked,
    )
    print(json.dumps(_json_safe(report), indent=2))


if __name__ == "__main__":
    main()
