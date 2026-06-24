import torch
from torch import nn

from fate_x.acpr_dynflow_swin.model import ACPRDynFlowSwinModel
from fate_x.acpr_dynflow_swin.types import SwinBackboneOutput
from fate_x.engine.acpr_dynflow_swin_data import SyntheticDynFlowSwinDataset, collate_dynflow_swin


class FakeBackbone(nn.Module):
    def forward(self, frames):
        bsz = frames.shape[0]
        predicate = torch.randn(bsz, 4, 7, 7, 256)
        final = torch.randn(bsz, 16, 7, 7, 256)
        return SwinBackboneOutput(
            predicate_grid=predicate,
            final_grid=final,
            temporal_global=final.mean(dim=(2, 3)),
            dense_final_tokens=final.reshape(bsz, 16 * 49, 256),
            forward_count=1,
        )


def test_model_uses_real_text_derived_nnpu_loss(monkeypatch):
    model = ACPRDynFlowSwinModel(
        {
            "paths": {"nnpu_rules": "configs/acpr_dynflow_swin_text_rules.yaml"},
            "model": {"backbone": {"use_tiny_test_backbone": True}},
        }
    )
    model.backbone = FakeBackbone()
    rows = [SyntheticDynFlowSwinDataset(length=2, image_size=32)[i] for i in range(2)]
    rows[0]["raw_action"] = "cars ahead stopped at a red light"
    rows[1]["raw_justification"] = "road is clear and there is no pedestrian"
    batch = collate_dynflow_swin(rows)
    output = model(batch)

    loss = output.loss_components["predicate_nnpu"]
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert output.diagnostics["nnpu_counts"]["positive"] > 0
    assert output.diagnostics["nnpu_counts"]["reliable_negative"] > 0
    assert output.diagnostics["nnpu_counts"]["unlabeled"] > 0
