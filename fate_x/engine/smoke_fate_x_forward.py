from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import torch

# ADAPT's bundled BERT utilities import boto3 for remote cache helpers. The
# FATE-X hook smoke does not use S3, so provide a tiny import stub when boto3 is
# not installed in the reproduction environment.
try:
    import boto3  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment compatibility shim
    import sys
    import types
    sys.modules["boto3"] = types.ModuleType("boto3")

from fate_x.models.video_token_reducer import VideoTokenReducer
from fate_x.models.temporal_evidence_memory import TemporalEvidenceMemory
from src.modeling.multitask_e2e_vid_swin_bert import MultitaskVideoTransformer


def build_shell_model(args: argparse.Namespace) -> MultitaskVideoTransformer:
    # Use the real ADAPT hook methods without constructing Swin/BERT. This isolates
    # the exact FATE-X reducer/memory + attention-mask resizing path inside ADAPT.
    model = MultitaskVideoTransformer.__new__(MultitaskVideoTransformer)
    torch.nn.Module.__init__(model)
    model.img_feature_dim = args.dim
    model.max_img_seq_length = args.max_img_seq_length
    model.fate_x_enabled = True
    model.video_token_reducer = args.video_token_reducer
    model.temporal_evidence_memory = args.temporal_evidence_memory
    model.fate_x_last_stats = {}
    model.fate_x_reducer = None if args.video_token_reducer == "none" else VideoTokenReducer(
        args.dim,
        keep_ratio=args.keep_ratio,
        num_summary_tokens=args.num_summary_tokens,
        min_tokens=args.min_tokens,
        mode=args.video_token_reducer,
    )
    model.fate_x_memory = None if args.temporal_evidence_memory == "none" else TemporalEvidenceMemory(args.dim, num_events=args.num_events)
    return model


def run_smoke(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    model = build_shell_model(args)
    tokens = torch.randn(args.batch_size, args.num_tokens, args.dim)
    text_len = args.text_len
    attention_mask = torch.ones(args.batch_size, text_len + args.max_img_seq_length, text_len + args.max_img_seq_length)
    kwargs = {"attention_mask": attention_mask}
    out = model._apply_fate_x_tokens(tokens, kwargs)
    result = {
        "input_shape": list(tokens.shape),
        "output_shape": list(out.shape),
        "attention_mask_shape": list(kwargs["attention_mask"].shape),
        "token_stats": model.fate_x_last_stats,
        "has_provenance": getattr(model, "fate_x_last_provenance", None) is not None,
    }
    expected_total = text_len + out.shape[1]
    if kwargs["attention_mask"].shape[-1] != expected_total:
        raise AssertionError(f"attention mask length mismatch: {kwargs['attention_mask'].shape[-1]} vs {expected_total}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the real ADAPT FATE-X token hook and attention-mask resize path.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--video_token_reducer", choices=["none", "topk_merge", "merge"], default="topk_merge")
    ap.add_argument("--temporal_evidence_memory", choices=["none", "queries"], default="queries")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_tokens", type=int, default=128)
    ap.add_argument("--max_img_seq_length", type=int, default=128)
    ap.add_argument("--text_len", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--keep_ratio", type=float, default=0.5)
    ap.add_argument("--num_summary_tokens", type=int, default=8)
    ap.add_argument("--min_tokens", type=int, default=16)
    ap.add_argument("--num_events", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    result = run_smoke(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()