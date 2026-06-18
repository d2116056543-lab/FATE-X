import torch
from fate_x.engine.backbone_output_utils import unpack_flowtrace_backbone_output
from fate_x.models.flowtrace_pmt_model import FlowTracePMTModel


def test_flowtrace_e2e_smoke_backward():
    model = FlowTracePMTModel(fine_dim=16, coarse_dim=32, dense_dim=32, state_dim=16, num_tracks=3, num_states=2)
    dense = torch.randn(2, 8, 32)
    fine = torch.randn(2, 16, 4, 6, 6)
    coarse = torch.randn(2, 32, 4, 3, 3)
    bundle = model(dense, fine, coarse)
    assert bundle.state_evidence_maps.shape[:3] == (2, 4, 2)
    loss = bundle.reason_state.pow(2).mean()
    loss.backward()
    assert model.tracks.track_queries.grad is not None


def test_flowtrace_backbone_output_unpack_supports_dict_and_checkpoint_tuple():
    final = torch.randn(1, 8, 2, 3, 3)
    fine = torch.randn(1, 16, 2, 2, 2)
    coarse = torch.randn(1, 32, 1, 1, 1)

    out_final, out_stages = unpack_flowtrace_backbone_output(
        {"final_tokens": final, "stages": [fine, coarse]}
    )
    assert out_final is final
    assert out_stages == [fine, coarse]

    out_final, out_stages = unpack_flowtrace_backbone_output((final, fine, coarse))
    assert out_final is final
    assert out_stages == [fine, coarse]
