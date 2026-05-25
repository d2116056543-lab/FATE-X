from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fate_x.engine.write_eval_artifacts import write_fate_x_eval_artifacts


def test_write_fate_x_eval_artifacts_creates_required_files(tmp_path):
    rows = [
        {
            "sample_id": "sample-1",
            "split": "validation",
            "gt_action": "go straight",
            "gt_justification": "traffic is clear",
            "pred_action": "go straight",
            "pred_justification": "traffic is clear",
            "generated_tokens": ["traffic", "is", "clear"],
            "token_logprobs": [-0.1, -0.2, -0.3],
            "phrase_hits": [{"phrase_type": "clear_road", "text": "traffic is clear"}],
            "phrase_scores": [{"phrase_type": "clear_road", "original_score": -0.2}],
            "token_stats": {"dense_visual_tokens": 32, "text_visual_tokens": 20, "control_visual_tokens": 32},
        }
    ]
    write_fate_x_eval_artifacts(tmp_path, 0, rows, run_manifest={"repo_name": "FATE-X", "is_smoke": True})
    epoch_dir = tmp_path / "epoch_000"
    for name in [
        "caption_metrics.json",
        "control_metrics.json",
        "token_stats.jsonl",
        "predictions.jsonl",
        "phrase_hits.jsonl",
        "phrase_scores.jsonl",
        "phrase_faithfulness_summary.json",
        "failure_cases.jsonl",
        "run_manifest.json",
    ]:
        assert (epoch_dir / name).exists(), name
    first_prediction = json.loads((epoch_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_prediction["token_stats"]["control_visual_tokens"] == 32


def test_phrase_score_cli_live_mode_fails_explicitly(tmp_path):
    out = tmp_path / "scores.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "fate_x.engine.generate_decoder_phrase_scores_from_model",
            "--mode",
            "live",
            "--output_jsonl",
            str(out),
            "--max_samples",
            "1",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "NotImplementedError" in combined or "live checkpoint" in combined
