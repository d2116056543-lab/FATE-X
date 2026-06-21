from __future__ import annotations

from types import SimpleNamespace

import torch

from fate_x.acpr_flow.model import ACPRFlowModel, ACPRFlowModelConfig
from fate_x.engine.train_acpr_flowcal_pp import summarize_traffic_flow_audit


def test_control_path_has_temporal_variation_for_prediction_delta_audit() -> None:
    model = ACPRFlowModel(ACPRFlowModelConfig(state_dim=8, text_hidden_dim=16, num_frames=6, formal_backbone=False))
    torch.manual_seed(7)
    state = torch.randn(4, 16)
    hidden = model._control_hidden_sequence(state, steps=6)
    assert not torch.allclose(hidden[:, 0], hidden[:, -1])

    bundle = SimpleNamespace(global_reason_state=state, reason_memory=torch.randn(4, 5, 16))
    ctrl = model.predict_control_from_bundle(bundle, steps=6)
    pred = ctrl["control_final_prediction"]
    assert pred.shape == (4, 6, 2)
    target = pred.clone()
    target[:, -1, 1] = target[:, 0, 1] + torch.linspace(-1.0, 1.0, 4)
    flow = torch.stack([torch.linspace(0.1, 0.9, 4), torch.linspace(0.9, 0.1, 4)], dim=1)
    audit = summarize_traffic_flow_audit(flow, None, pred, target, flow_factor_names=["up", "down"], signal_names=["course", "speed"])
    assert audit["delta_stats"]["speed"]["pred_delta_zero_variance"] is False
    assert audit["flow_factors"]["up"]["pred_speed_delta_corr_reason"] == "ok"
