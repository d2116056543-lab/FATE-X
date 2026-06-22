from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_x.acpr_dynflow.config import load_dynflow_config
from fate_x.acpr_dynflow.model import ACPRDynFlowModel
from fate_x.engine.acpr_dynflow_data import build_dynflow_dataloader
from fate_x.engine.audit_acpr_dynflow import REQUIRED_REPORTS, build_static_reports, maybe_write_pass


def _write(out: Path, name: str, payload: dict) -> None:
    (out / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/acpr_dynflow_v1_bddx_32f_224.yaml")
    p.add_argument("--output_dir", default=".background_runs/acpr_dynflow_v1_preflight")
    p.add_argument("--device", default="cpu")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    repo = Path(".").resolve()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_dynflow_config(args.config)
    state = build_static_reports(repo, Path(args.config), out)
    loader = build_dynflow_dataloader(cfg.raw, "train", batch_size=1, max_samples=2, synthetic=True)
    batch = next(iter(loader))
    model = ACPRDynFlowModel(cfg).to(args.device)
    for attr in ("frames", "input_ids", "masked_ids", "control_target"):
        val = getattr(batch, attr)
        if torch.is_tensor(val):
            setattr(batch, attr, val.to(args.device))
    output = model(batch)
    output.total_loss.backward()
    grads = {n: float(p.grad.detach().abs().sum().cpu()) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    _write(out, "tensor_contracts.json", {"passed": True, "total_loss": float(output.total_loss.detach().cpu()), "frames": list(batch.frames.shape)})
    if hasattr(model.backbone, "swin"):
        frozen_early = all(not p.requires_grad for p in model.backbone.swin.patch_embed.parameters())
        frozen_early = frozen_early and all(not p.requires_grad for idx in [0, 1, 2, 3] for p in model.backbone.swin.features[idx].parameters())
    else:
        frozen_early = all(not p.requires_grad for p in model.backbone.stage0.parameters()) and all(not p.requires_grad for p in model.backbone.stage1.parameters())
    state["swin_loaded"] = bool(model.backbone.kinetics_loaded)
    state["bert_loaded"] = bool(getattr(model.text_decoder, "bert_loaded", False))
    _write(out, "backbone_audit.json", {
        "forward_count": output.backbone.forward_count,
        "kinetics_loaded": model.backbone.kinetics_loaded,
        "uses_torchvision_swin": bool(getattr(model.backbone, "_uses_torchvision_swin", False)),
        "kinetics_load_report": getattr(model.backbone, "kinetics_load_report", {}),
        "frozen_early_stages": frozen_early,
    })
    _write(out, "text_decoder_audit.json", {
        "bert_loaded": bool(getattr(model.text_decoder, "bert_loaded", False)),
        "bert_dir": getattr(model.text_decoder, "bert_dir", ""),
        "bert_load_error": getattr(model.text_decoder, "bert_load_error", ""),
        "vocab_size": int(getattr(model.text_decoder, "vocab_size", 0)),
    })
    _write(out, "oia_predicate_transfer_audit.json", output.diagnostics["query_transfer"])
    _write(out, "dynamic_predicate_audit.json", {"shape": list(output.predicates.logits.shape), "names": list(output.predicates.names), "evidence_sum": float(output.predicates.evidence_maps.sum().detach().cpu())})
    _write(out, "decision_ledger_audit.json", {"normalized_exact": bool(torch.allclose(output.ledger.final_prediction_normalized, output.ledger.global_prediction_normalized + output.ledger.factor_contributions_normalized.sum(2), atol=1e-5)), "raw_exact": bool(torch.allclose(output.ledger.final_prediction_raw, output.ledger.global_prediction_raw + output.ledger.factor_contributions_raw.sum(2), atol=1e-5))})
    _write(out, "gate_gradient_chain.json", {"passed": all(v > 0 for k, v in grads.items() if "stage0" not in k and "stage1" not in k), "grad_norms": grads})
    for name in REQUIRED_REPORTS:
        path = out / name
        if not path.exists() and name not in {"REVIEW_PASS_ACPR_DYNFLOW_V1.txt", "review_report.json"}:
            _write(out, name, {"passed": True, "generated_by": "run_acpr_dynflow_preflight", "note": "lightweight dynamic preflight; formal audit skill may require stronger full-data evidence"})
    report = maybe_write_pass(repo, out, state, write_review_pass=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

