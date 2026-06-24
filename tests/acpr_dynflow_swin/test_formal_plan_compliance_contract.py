from __future__ import annotations

import importlib
import inspect


def test_formal_backbone_uses_repository_video_swin_not_conv_fallback():
    module = importlib.import_module("fate_x.acpr_dynflow_swin.video_swin_backbone")
    source = inspect.getsource(module)
    assert "get_swin_model" in source or "myVideoSwin" in source
    assert "src.modeling.load_swin" in source
    assert "Conv2d" not in source
    assert "fallback" not in source.lower()


def test_motion_transformer_is_bert_base_capacity_12_layer_model():
    module = importlib.import_module("fate_x.acpr_dynflow_swin.query_motion_transformer")
    signature = inspect.signature(module.QueryMotionTransformer)
    assert signature.parameters["hidden_dim"].default == 768
    assert signature.parameters["num_layers"].default == 12
    assert signature.parameters["nhead"].default == 12
    assert signature.parameters["intermediate_size"].default == 3072


def test_text_decoder_uses_adapt_bert_autoregressive_captioning_path():
    module = importlib.import_module("fate_x.acpr_dynflow_swin.text_decoder")
    source = inspect.getsource(module)
    assert "BertForImageCaptioning" in source
    assert ".generate(" in source
    assert "generated_action=None" not in source
    assert "generated_explanation=None" not in source


def test_signal_codec_module_exposes_train_stats_and_official_metrics():
    module = importlib.import_module("fate_x.acpr_dynflow_swin.signal_codec")
    assert hasattr(module, "BDDXSignalCodec")
    codec = module.BDDXSignalCodec(signal_names=("course", "speed"))
    assert hasattr(codec, "fit")
    assert hasattr(codec, "encode")
    assert hasattr(codec, "decode")
    assert hasattr(codec, "official_metrics")
