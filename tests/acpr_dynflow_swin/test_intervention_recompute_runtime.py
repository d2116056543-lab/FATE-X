import torch
from torch import nn

from fate_x.acpr_dynflow_swin.interventions import InterventionSpec, run_intervention
from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.acpr_dynflow_swin.types import SwinBackboneOutput
from fate_x.engine.acpr_dynflow_swin_data import SyntheticDynFlowSwinDataset, collate_dynflow_swin


class CountingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, frames):
        self.calls += 1
        bsz = frames.shape[0]
        weights = torch.arange(frames.shape[1], dtype=frames.dtype).view(1, -1, 1, 1, 1)
        summary = (frames * weights).mean()
        predicate = summary.expand(bsz, 4, 7, 7, 256).clone()
        final = summary.expand(bsz, 16, 7, 7, 256).clone()
        return SwinBackboneOutput(
            predicate_grid=predicate,
            final_grid=final,
            temporal_global=final.mean(dim=(2, 3)),
            dense_final_tokens=final.reshape(bsz, 16 * 49, 256),
            forward_count=self.calls,
        )


def _batch():
    rows = [SyntheticDynFlowSwinDataset(length=1, image_size=32)[0]]
    rows[0]["raw_action"] = "cars ahead stopped"
    return collate_dynflow_swin(rows)


def test_temporal_reverse_reruns_from_frames_and_records_trace():
    model = ACPRDynFlowSwinModel({"paths": {"nnpu_rules": "configs/acpr_dynflow_swin_text_rules.yaml"}})
    model.backbone = CountingBackbone()
    base = model(_batch())
    counter = run_intervention(model, _batch(), InterventionSpec("temporal_reverse"))
    assert model.backbone.calls == 2
    assert counter.diagnostics["intervention"]["earliest_layer"] == "frames"
    assert "backbone" in counter.diagnostics["intervention"]["rerun_layers"]
    assert not torch.allclose(base.ledger.final_prediction_normalized, counter.ledger.final_prediction_normalized)


def test_factor_off_reruns_downstream_from_traffic():
    model = ACPRDynFlowSwinModel({"paths": {"nnpu_rules": "configs/acpr_dynflow_swin_text_rules.yaml"}})
    model.backbone = CountingBackbone()
    counter = run_intervention(model, _batch(), InterventionSpec("factor_off", factor_index=0))
    trace = counter.diagnostics["intervention"]
    assert trace["earliest_layer"] == "traffic"
    assert trace["rerun_layers"] == ["ledger", "text"]
    assert counter.traffic.factor_tokens_native[:, :, 0].abs().sum().item() == 0
