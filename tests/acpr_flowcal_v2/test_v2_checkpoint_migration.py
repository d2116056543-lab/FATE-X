from fate_x.acpr_flow_v2.config import FlowCalV2Config
from fate_x.acpr_flow_v2.model import ACPRFlowCalV2Model
from fate_x.engine.train_acpr_flowcal_v2 import CheckpointMigratorV1ToV2


def test_checkpoint_migrator_rejects_tmp_checkpoint():
    model = ACPRFlowCalV2Model(FlowCalV2Config(hidden_dim=8, text_vocab_size=11))
    try:
        CheckpointMigratorV1ToV2().migrate("bad.tmp", model)
    except ValueError:
        return
    raise AssertionError("tmp checkpoint must be rejected")
