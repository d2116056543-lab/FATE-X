import torch

from fate_x.acpr_flow_v2.axis_aware_control_adapter import AxisAwareReasonControlAdapter
from fate_x.acpr_flow_v2.types import SemanticReasonMemory


def test_zeroed_control_adapter_residual_preserves_base():
    adapter = AxisAwareReasonControlAdapter(hidden_dim=8, max_residual_std_fraction=0.0)
    base = torch.randn(2, 32, 2)
    hidden = torch.randn(2, 32, 8)
    memory = SemanticReasonMemory(
        values=torch.randn(2, 54, 8),
        mask=torch.ones(2, 54, dtype=torch.bool),
        confidence=torch.ones(2, 54),
        names=tuple(str(i) for i in range(54)),
        type_ids=torch.zeros(54, dtype=torch.long),
        axis_ids=torch.ones(54, dtype=torch.long) * 3,
        evidence_maps=torch.zeros(2, 1, 54, 2, 2),
        lineage=[],
        semantic_state=torch.randn(2, 8),
    )
    out = adapter(base, hidden, memory, None)
    assert torch.allclose(out.final_prediction, base)
