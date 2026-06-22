import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from _v2_contract_smoke import make_reason_memory
from fate_x.acpr_flow_v2.axis_aware_control_adapter import AxisAwareReasonControlAdapter


def test_axis_control_adapter_outputs_bounded_residual():
    base = torch.zeros(1, 32, 2)
    hidden = torch.randn(1, 32, 8)
    out = AxisAwareReasonControlAdapter(hidden_dim=8, max_residual_std_fraction=0.2)(base, hidden, make_reason_memory(dim=8), None)
    assert out.final_prediction.shape == base.shape
    assert out.residual.abs().max() <= 0.2 + 1e-6
