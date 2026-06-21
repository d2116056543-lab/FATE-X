from __future__ import annotations

import json
from pathlib import Path

from fate_x.engine.acpr_action_text_eval import classify_direction_action, classify_speed_action, evaluate_action_text_decisions


def test_classify_speed_action_keywords():
    assert classify_speed_action("the car slows down to a stop") == "stop"
    assert classify_speed_action("the vehicle is braking for traffic") == "slow"
    assert classify_speed_action("the car accelerates after the light turns green") == "accelerate"
    assert classify_speed_action("the car is moving at a steady speed") == "maintain"


def test_classify_direction_action_keywords():
    assert classify_direction_action("the car turns left at the intersection") == "left"
    assert classify_direction_action("the vehicle is turning right") == "right"
    assert classify_direction_action("the car continues straight") == "straight"


def test_evaluate_action_text_decisions_from_adapt_tsv(tmp_path: Path):
    gt = tmp_path / "testing.caption.tsv"
    pred = tmp_path / "pred.BDDX.testing_32frames.beam1.max15.tsv"
    gt.write_text(
        "a\t" + json.dumps([{"action": "The car slows down to a stop"}]) + "\n"
        "b\t" + json.dumps([{"action": "The car accelerates"}]) + "\n"
        "c\t" + json.dumps([{"action": "The car turns left"}]) + "\n",
        encoding="utf-8",
    )
    pred.write_text(
        "a\t" + json.dumps([{"caption": "the car is stopped"}]) + "\t[]\n"
        "b\t" + json.dumps([{"caption": "the car is slowing down"}]) + "\t[]\n"
        "c\t" + json.dumps([{"caption": "the car turns left"}]) + "\t[]\n",
        encoding="utf-8",
    )

    report = evaluate_action_text_decisions(pred, gt)

    assert report["speed_decision"]["evaluated_count"] == 2
    assert report["speed_decision"]["accuracy"] == 0.5
    assert report["direction_decision"]["evaluated_count"] == 1
    assert report["direction_decision"]["accuracy"] == 1.0
