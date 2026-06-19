import torch

from fate_x.acpr_flow.interventions import InterventionSpec
from fate_x.acpr_flow.model import ACPRFlowModel


def test_intervention_reruns_downstream_and_changes_reason_memory():
    model = ACPRFlowModel()
    frames = torch.randn(1, 32, 3, 64, 64)
    base = model.build_bundle(frames)
    counter = model.build_bundle(frames, intervention=InterventionSpec(kind="temporal_reverse"))
    assert not torch.allclose(base.reason_memory, counter.reason_memory)
