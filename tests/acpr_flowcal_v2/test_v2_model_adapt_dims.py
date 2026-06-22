from fate_x.acpr_flow_v2.config import FlowCalV2Config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model


def test_model_uses_adapt_dense_dim_for_motion_and_state_dim_for_reasoning():
    cfg = FlowCalV2Config(use_real_video_swin=False, state_dim=16, hidden_dim=16, adapt_feature_dim=512, text_hidden_dim=32)
    model = ACPRFlowCalV2Model(cfg)
    assert model.video.fc.out_features == 512
    assert model.video.reason_proj.out_features == 16
    assert model.motion.input_dim == 512
