from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from _v2_contract_smoke import run_tiny_forward
from fate_x.acpr_flow_v2.interventions import FlowCalV2InterventionEngine, InterventionSpecV2


def test_intervention_engine_reruns_from_visual_layer():
    out = run_tiny_forward()
    changed = FlowCalV2InterventionEngine().rerun_from_visual(out.bundle, InterventionSpecV2(kind="flow_off"))
    assert changed is not out.bundle

def test_flow_off_intervention_changes_model_control_output():
    from fate_x.acpr_flow_v2.config import FlowCalV2Config
    from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
    from fate_x.acpr_flow_v2.types import FlowCalV2Batch
    import torch

    model = ACPRFlowCalV2Model(FlowCalV2Config(hidden_dim=32, state_dim=32, text_vocab_size=101))
    batch = FlowCalV2Batch(
        frames=torch.randn(1, 32, 3, 224, 224),
        input_ids=torch.randint(0, 101, (1, 30)),
        masked_pos=torch.tensor([[1, 2]]),
        masked_ids=torch.randint(0, 101, (1, 2)),
        car_info=torch.randn(1, 2, 32),
    )
    base = model(batch)
    cf = model(batch, intervention=InterventionSpecV2(kind='all_flow_off'))
    delta = (base.control_final_prediction - cf.control_final_prediction).abs().mean().item()
    assert delta > 0
