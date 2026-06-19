import inspect

from src.layers.bert.modeling_bert import BertForImageCaptioning


def test_acpr_seca_hook_is_in_bert_captioning_pre_lm_path():
    src = inspect.getsource(BertForImageCaptioning.encode_forward)
    assert "acpr_temporal_seca" in src
    assert "acpr_flow_bundle.reason_memory" in src
    assert "outputs = (hidden,) + outputs[1:]" in src
