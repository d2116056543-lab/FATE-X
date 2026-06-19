import json
from pathlib import Path

from fate_x.engine import run_acpr_flowcal_preflight_gates


def test_preflight_gate_runner_writes_required_gate_artifacts(tmp_path, monkeypatch):
    def fake_train_formal(config, output_dir, device, max_steps, batch_size, epochs, load_pretrained_backbone=True):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for step in range(max_steps):
            rows.append(
                json.dumps(
                    {
                        "global_step": step + 1,
                        "loss": float(max_steps - step),
                        "frames_shape": [batch_size, 32, 3, 224, 224],
                        "loss_components": {
                            "action_text": float(max_steps - step),
                            "explanation_text": float(max_steps - step),
                            "control": float(max_steps - step),
                            "hardpair_active_pair_rate": 0.5,
                        },
                    }
                )
            )
        (out / "metrics_summary.jsonl").write_text("\n".join(rows), encoding="utf-8")
        (out / "checkpoint_latest.pth").write_text("stub", encoding="utf-8")

    def fake_run_audit(config, output_dir, device="cpu", write_review_pass=False):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return {
            "direct_image_shape": [1, 32, 3, 224, 224],
            "gradient_report": {
                "predicate_query": 1.0,
                "flow_query": 1.0,
                "reason_memory": 1.0,
                "seca_gate_action": 1.0,
                "control_gate": 1.0,
                "hardpair_projection": 1.0,
            },
            "intervention_delta": 0.25,
        }

    monkeypatch.setattr(run_acpr_flowcal_preflight_gates, "train_formal", fake_train_formal)
    monkeypatch.setattr(run_acpr_flowcal_preflight_gates, "run_audit", fake_run_audit)

    report = run_acpr_flowcal_preflight_gates.run_preflight_gates(
        config="cfg.yaml",
        output_dir=tmp_path,
        device="cuda",
        gate_b_steps=8,
        gate_d_steps=4,
    )

    for name in [
        "gate_b_direct_image_8step_smoke.json",
        "gate_c_gradient_chain_report.json",
        "gate_d_mechanism_overfit_128_report.json",
        "gate_e_temporal_necessity_report.json",
        "gate_f_real_intervention_report.json",
    ]:
        assert (tmp_path / name).exists()
    assert report["gate_b"]["step_count"] == 8
    assert report["gate_c"]["all_required_gradients_nonzero"] is True
    assert report["gate_f"]["state_off_delta_nonzero"] is True
