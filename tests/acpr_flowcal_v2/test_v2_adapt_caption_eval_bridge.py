from __future__ import annotations

import json

from fate_x.engine.adapt_caption_eval_bridge import run_adapt_sep_caption_eval, write_adapt_sep_caption_tsv


class _DatasetNoCaption:
    yaml_file = "BDDX/testing_32frames.yaml"


class _LoaderNoCaption:
    dataset = _DatasetNoCaption()


def test_adapt_sep_caption_tsv_has_description_and_explanation_columns(tmp_path):
    out = write_adapt_sep_caption_tsv(
        [{"img_key": "vid_001", "description": "car slows down", "explanation": "because traffic is ahead"}],
        tmp_path / "pred.BDDX.test.tsv",
    )

    line = out.read_text(encoding="utf-8").strip()
    parts = line.split("\t")
    assert parts[0] == "vid_001"
    assert json.loads(parts[1])[0]["caption"] == "car slows down"
    assert json.loads(parts[2])[0]["caption"] == "because traffic is ahead"


def test_adapt_caption_eval_reports_blocker_without_reference_caption(tmp_path):
    metrics = run_adapt_sep_caption_eval(
        [{"img_key": "vid_001", "description": "d", "explanation": "e"}],
        _LoaderNoCaption(),
        tmp_path,
    )

    assert metrics["text_metrics_available"] is False
    assert "does not expose" in metrics["text_metrics_blocker"]
    assert "CIDEr_des" not in metrics
    assert "CIDEr_exp" not in metrics
