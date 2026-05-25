from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from fate_x.engine.generate_decoder_phrase_scores import score_decoder_phrase_rows
from fate_x.engine.write_eval_artifacts import write_fate_x_eval_artifacts


MASKS = {
    None: None,
    "topk": "topk",
    "evidence_only": "evidence_only",
    "random": "random",
}


def _require_generation(record: dict[str, Any], mask_name: str) -> tuple[str, list[str], list[float]]:
    text = str(record.get("prediction") or record.get("caption") or record.get("text") or "")
    tokens = [str(x) for x in record.get("tokens", [])]
    logprobs = record.get("token_logprobs")
    if not text or not tokens or logprobs is None:
        raise ValueError(f"Model output for mask={mask_name} must contain prediction/text, tokens, and token_logprobs")
    return text, tokens, [float(x) for x in logprobs]


def collect_phrase_scores_from_model(
    model: Any,
    samples: Iterable[dict[str, Any]],
    *,
    topk_ratio: float = 0.10,
    mask_strategy: str = "zero",
) -> list[dict[str, Any]]:
    """Collect model-in-loop phrase perturbation rows.

    The model contract is deliberately small so this can be tested without
    loading ADAPT: it must expose ``generate_with_logprobs(sample, mask=...)``.
    A real ADAPT wrapper should implement mask values ``topk``,
    ``evidence_only`` and ``random`` using zero/background replacement of video
    evidence rather than deleting tokens and changing tensor shapes.
    """
    rows: list[dict[str, Any]] = []
    for sample in samples:
        base = model.generate_with_logprobs(sample, mask=None)
        text, tokens, logprobs = _require_generation(base, "none")
        row: dict[str, Any] = {
            "id": sample.get("id") or sample.get("sample_id") or sample.get("file_name"),
            "prediction": text,
            "tokens": tokens,
            "token_logprobs": logprobs,
            "topk_ratio": float(topk_ratio),
            "mask_strategy": mask_strategy,
        }
        perturbation_fields = {
            "topk": "topk_masked_token_logprobs",
            "evidence_only": "evidence_only_token_logprobs",
            "random": "random_masked_token_logprobs",
        }
        for mask, field in perturbation_fields.items():
            perturbed = model.generate_with_logprobs(sample, mask=mask)
            _, _, perturbed_logprobs = _require_generation(perturbed, mask)
            row[field] = perturbed_logprobs
        rows.append(row)
    scored, _ = score_decoder_phrase_rows(rows)
    return scored


class JsonlReplayModel:
    """Small utility for smoke tests with precomputed model outputs.

    Real checkpoint inference must be supplied by an ADAPT wrapper. This class
    exists only so the CLI can validate schema and perturbation scoring without
    fabricating scores.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def generate_with_logprobs(self, sample: dict[str, Any], mask=None) -> dict[str, Any]:
        idx = int(sample["row_index"])
        row = self.rows[idx]
        out = {"prediction": row.get("prediction") or row.get("caption") or row.get("text"), "tokens": row.get("tokens", [])}
        if mask is None:
            out["token_logprobs"] = row.get("token_logprobs")
        elif mask == "topk":
            out["token_logprobs"] = row.get("topk_masked_token_logprobs")
        elif mask == "evidence_only":
            out["token_logprobs"] = row.get("evidence_only_token_logprobs")
        elif mask == "random":
            out["token_logprobs"] = row.get("random_masked_token_logprobs")
        else:
            raise ValueError(f"Unsupported replay mask: {mask}")
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate model-in-loop BDD-X phrase perturbation scores.")
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--eval_model_dir", default="", help="ADAPT/FATE-X checkpoint directory. Real wrapper integration is required for live inference.")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--val_yaml", default="")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--artifact_output_dir", default="", help="Optional FATE-X eval artifact directory for epoch_000 schema output.")
    ap.add_argument("--max_samples", type=int, default=32)
    ap.add_argument("--beam_size", type=int, default=1)
    ap.add_argument("--mask_strategy", choices=["zero", "background", "mask_token"], default="zero")
    ap.add_argument("--topk_ratio", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--replay_decoder_jsonl", default="", help="Schema smoke path with precomputed live-model rows.")
    args = ap.parse_args()

    if args.mode == "live":
        raise NotImplementedError(
            "live checkpoint-dependent phrase faithfulness requires an ADAPT/FATE-X "
            "decoder wrapper that exposes generate_with_logprobs(sample, mask=...). "
            "Replay mode is only a schema smoke and is not a live faithfulness claim."
        )
    if not args.replay_decoder_jsonl:
        raise SystemExit(
            "Replay mode needs --replay_decoder_jsonl. Live ADAPT checkpoint phrase scoring needs a repository-specific decoder wrapper. "
            "Use --replay_decoder_jsonl for schema smoke, or implement generate_with_logprobs "
            "around the trained ADAPT/FATE-X checkpoint before claiming checkpoint-dependent faithfulness."
        )
    replay_rows = [json.loads(line) for line in Path(args.replay_decoder_jsonl).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if args.max_samples > 0:
        replay_rows = replay_rows[: args.max_samples]
    model = JsonlReplayModel(replay_rows)
    samples = [{"row_index": i, "id": replay_rows[i].get("id", i)} for i in range(len(replay_rows))]
    scored = collect_phrase_scores_from_model(model, samples, topk_ratio=args.topk_ratio, mask_strategy=args.mask_strategy)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.summary_json:
        _, summary = score_decoder_phrase_rows(scored)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.artifact_output_dir:
        write_fate_x_eval_artifacts(args.artifact_output_dir, 0, scored, run_manifest={"repo_name": "FATE-X", "mode": args.mode, "is_smoke": True})
    print(json.dumps({"event": "fate_x_phrase_model_loop_rows", "rows": len(scored), "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
