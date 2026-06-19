import torch

from fate_x.acpr_flow.model import ACPRFlowModel
from fate_x.acpr_flow.types import ACPRFlowTrainOutput


def test_model_returns_typed_named_output_contract():
    out = ACPRFlowModel()(torch.randn(1, 32, 3, 64, 64))
    assert isinstance(out, ACPRFlowTrainOutput)
    assert out.baseline_masked_logits.shape[-1] == out.enhanced_masked_logits.shape[-1]
    assert out.control_base_prediction.shape == out.control_final_prediction.shape == (1, 32, 2)
    assert out.bundle.reason_memory.shape[1] == 46
