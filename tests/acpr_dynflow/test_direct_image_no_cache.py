from fate_x.engine.acpr_dynflow_data import SyntheticDynFlowDataset, collate_dynflow

def test_direct_image_shape():
    b=collate_dynflow([SyntheticDynFlowDataset(length=1)[0]])
    assert list(b.frames.shape) == [1,32,3,224,224]

