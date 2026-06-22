import torch

from fate_x.acpr_flow_v2.config import ACPRFlowCalV2Config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.acpr_flow_v2.types import FlowCalV2Batch


def test_v2_model_consumes_direct_frames_and_returns_required_outputs():
    cfg = ACPRFlowCalV2Config(hidden_dim=32, text_vocab_size=17, num_frames=32)
    model = ACPRFlowCalV2Model(cfg)
    batch = FlowCalV2Batch(
        frames=torch.randn(2, 32, 3, 224, 224),
        input_ids=torch.randint(0, 17, (2, 30)),
        attention_mask=torch.ones(2, 30, dtype=torch.long),
        masked_pos=torch.tensor([[1, 2], [3, 4]]),
        masked_ids=torch.randint(0, 17, (2, 2)),
        car_info=torch.randn(2, 2, 32),
        sample_ids=["a", "b"],
    )

    out = model(batch, stage="R")
    assert out.total_loss.ndim == 0
    assert out.text_logits.shape[:2] == (2, 30)
    assert out.control_pred.shape == (2, 32, 2)
    assert out.bundle.local_transport_probs.shape[0] == 2
    assert "traffic_density" in out.bundle.diagnostics
    for key in (
        "traffic_density_per_sample",
        "traffic_queue_per_sample",
        "traffic_motion_per_sample",
        "traffic_stopped_per_sample",
        "traffic_coherence_per_sample",
        "traffic_transport_shift_per_sample",
        "traffic_transport_dustbin_per_sample",
    ):
        assert key in out.bundle.diagnostics
        assert out.bundle.diagnostics[key].shape == (2,)
