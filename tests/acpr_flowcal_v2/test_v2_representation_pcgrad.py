import torch

from fate_x.acpr_flow_v2.semantic_gradient_firewall import representation_pcgrad_surrogate
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from _v2_contract_smoke import make_reason_memory


def test_representation_pcgrad_surrogate_is_finite_and_centered():
    sem = torch.tensor(1.0, requires_grad=True)
    ctrl = torch.tensor(0.5, requires_grad=True)
    surr, diag = representation_pcgrad_surrogate(make_reason_memory(dim=8), sem, ctrl)
    assert torch.isfinite(surr)
    assert abs(float(surr.detach())) < 1e-4
    assert "semantic_grad_norm" in diag
