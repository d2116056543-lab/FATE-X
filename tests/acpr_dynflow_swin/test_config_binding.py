
def test_config_loads_and_core_fields_are_consumed():
    from fate_x.acpr_dynflow_swin.config import load_config, build_config_consumer_manifest

    cfg = load_config("configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml")
    manifest = build_config_consumer_manifest(cfg)
    for dotted in [
        "model.semantic_consolidation.slot_names",
        "model.traffic.pattern_dilations",
        "model.traffic.response_lags",
        "model.ledger.non_degradation_margin_normalized",
        "optimization.learning_rates.video_swin_backbone",
        "memory_throughput_probe.hard_peak_reserved_limit_gib",
    ]:
        assert dotted in manifest
        assert manifest[dotted]["consumer"]
