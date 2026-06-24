import json
from pathlib import Path

import torch


def test_trainer_builds_warmup_linear_scheduler_and_accumulation_state(tmp_path):
    from fate_x.engine.train_acpr_dynflow_swin import (
        build_linear_warmup_decay_scheduler,
        build_training_state,
        atomic_save_checkpoint,
    )

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = build_linear_warmup_decay_scheduler(
        optimizer,
        total_optimizer_steps=10,
        warmup_ratio=0.2,
        min_lr_ratio=0.0,
    )
    lrs = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == 0.5
    assert lrs[1] == 1.0
    assert lrs[-1] < lrs[1]

    state = build_training_state(
        epoch=3,
        global_step=17,
        optimizer_step=4,
        gradient_accumulation_steps=16,
        best_records={"text": {"epoch": 1}},
        signal_codec={"signal_names": ["course", "speed"]},
        config={"optimization": {"precision": "bf16"}},
    )
    assert state["gradient_accumulation_steps"] == 16
    assert state["rng_state"]["torch_cpu"] is not None
    assert state["best_records"]["text"]["epoch"] == 1

    ckpt_path = tmp_path / "checkpoint_latest.pth"
    atomic_save_checkpoint(
        ckpt_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            **state,
        },
    )
    assert ckpt_path.exists()
    assert not (tmp_path / "checkpoint_latest.pth.tmp").exists()
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "optimizer" in loaded
    assert "scheduler" in loaded
    assert loaded["optimizer_step"] == 4


def test_trainer_loads_adapt_reference_metrics_from_real_json(tmp_path):
    from fate_x.engine.train_acpr_dynflow_swin import load_adapt_reference_metrics

    des = tmp_path / "des.json"
    exp = tmp_path / "exp.json"
    des.write_text(json.dumps({"CIDEr": 2.0, "Bleu_4": 0.3}), encoding="utf-8")
    exp.write_text(json.dumps({"CIDEr": 0.75, "Bleu_4": 0.1}), encoding="utf-8")
    cfg = {
        "paths": {
            "adapt_reference_description_metrics": str(des),
            "adapt_reference_explanation_metrics": str(exp),
        },
        "evaluation": {
            "adapt_reference_control": {
                "speed_RMSE": 2.68,
                "course_RMSE": 5.87,
            }
        },
    }
    ref = load_adapt_reference_metrics(cfg)
    assert ref["CIDEr_description"] == 2.0
    assert ref["CIDEr_explanation"] == 0.75
    assert ref["CIDEr_sum"] == 2.75
    assert ref["speed_RMSE"] == 2.68
    assert ref["course_RMSE"] == 5.87
