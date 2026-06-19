import json
from pathlib import Path

from fate_x.engine import probe_acpr_flowcal_memory
from fate_x.engine.supervise_acpr_flowcal_foreground import write_foreground_supervisor_smoke


def test_memory_probe_runs_candidates_and_writes_selection(tmp_path, monkeypatch):
    calls = []

    def fake_train_formal(
        config,
        output_dir,
        device,
        max_steps,
        batch_size,
        epochs,
        gradient_accumulation_steps=1,
        checkpoint_every_steps=500,
        load_pretrained_backbone=True,
    ):
        calls.append(
            {
                "config": config,
                "output_dir": str(output_dir),
                "device": device,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "epochs": epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            }
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "metrics_summary.jsonl").write_text('{"loss": 1.0}\n', encoding="utf-8")

    monkeypatch.setattr(probe_acpr_flowcal_memory, "train_formal", fake_train_formal)

    report = probe_acpr_flowcal_memory.run_memory_probe(
        config="cfg.yaml",
        output_dir=tmp_path,
        device="cuda",
        candidates=[{"batch_size": 2, "gradient_accumulation_steps": 32}],
        warmup_steps=1,
        measured_steps=2,
    )

    assert calls and calls[0]["batch_size"] == 2
    assert calls[0]["max_steps"] == 3
    assert calls[0]["gradient_accumulation_steps"] == 32
    assert report["selected"]["effective_batch"] == 64
    selection = json.loads((tmp_path / "memory_probe_selection.json").read_text(encoding="utf-8"))
    assert selection["selected"]["batch_size"] == 2
    assert selection["direct_image_training"] is True
    assert (tmp_path / "memory_probe.json").exists()


def test_foreground_supervisor_writes_attached_proof(tmp_path):
    artifact = write_foreground_supervisor_smoke(
        output_dir=tmp_path,
        command=["python", "-m", "fate_x.engine.train_acpr_flowcal_pp"],
        heartbeat_seconds=60,
    )
    data = json.loads(Path(artifact).read_text(encoding="utf-8"))
    assert data["attached_foreground"] is True
    assert data["detached_process"] is False
    assert data["heartbeat_seconds"] == 60
    assert "python" in data["command"][0]
